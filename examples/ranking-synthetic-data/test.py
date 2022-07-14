import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data import IterableDataset


def generate_synthetic_LTR_data(
    majority_proportion=0.8, num_queries=100, num_docs_per_query=10, seed=0
):
    num_items = num_queries * num_docs_per_query
    X = np.random.uniform(0, 3, size=(num_items, 2)).astype(np.float32)
    relevance = X[:, 0] + X[:, 1]

    # i don't know why but the "fair policy" paper clips the values between 0 and 5
    relevance = np.clip(relevance, 0.0, 5.0)
    majority_status = np.random.choice(
        [True, False], size=num_items, p=[majority_proportion, 1 - majority_proportion]
    )
    X[~majority_status, 1] = 0
    return [
        {"X": X[i], "relevance": relevance[i], "majority_status": majority_status[i]}
        for i in range(num_items)
    ]


class QueryIterableDataset(IterableDataset):
    def __init__(self, items_dataset, shuffle, query_size):
        self.dataset = items_dataset
        self.query_size = query_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            idx = self._infinite_indices()
            query = [self.dataset[i] for i in next(idx)]
            query = torch.utils.data.default_collate(query)
            yield query

    def _infinite_indices(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = 0 if worker_info is None else worker_info.id
        g = torch.Generator()
        g.manual_seed(seed)
        while True:
            if self.shuffle:
                idx = (torch.randperm(len(self.dataset))[: self.query_size]).tolist()
                yield idx


num_docs_per_query = 10
num_queries = 100
dataset_train = generate_synthetic_LTR_data(
    num_queries=num_queries, num_docs_per_query=num_docs_per_query
)
dataloader = torch.utils.data.DataLoader(
    QueryIterableDataset(dataset_train, True, 10), num_workers=2, batch_size=2
)


# we perform a logistic regression on the dataset to build a sensitive direction
from sklearn.linear_model import LogisticRegression

all_data = torch.utils.data.default_collate(dataset_train)
x = all_data["X"]
majority_status = all_data["majority_status"]
LR = LogisticRegression(C=100).fit(x, majority_status)
sens_directions = torch.tensor(LR.coef_, dtype=torch.float32).T
print("sensitive directions", sens_directions)


# with this sensitive direction we can build a mahalanobis distance by passing a set of vectors
from inFairness.distances import SensitiveSubspaceDistance

sigma = SensitiveSubspaceDistance().compute_projection_complement(sens_directions)
print("sigma", sigma)


# vectorized version of the mahalanobis distance
from functorch import vmap


def md(x, y, sigma):
    """
    computes the mahalanobis distance between 2 vectors of D dimensions:

    .. math:: MD = (x - y) \\Sigma (x - y)^{'}
    """
    diff = x - y
    return torch.einsum("i,ij,j", diff, sigma, diff)


md1 = vmap(md, in_dims=(None, 0, None))
md2 = vmap(md1, in_dims=(0, None, None))
vect_mahalanobis_distance = vmap(md2, in_dims=(0, 0, None))


from geomloss import SamplesLoss

x, y = next(iter(dataloader))["X"], next(iter(dataloader))["X"]
wasserstein_distance = SamplesLoss(
    "sinkhorn", cost=lambda x, y: vect_mahalanobis_distance(x, y, sigma)
)
x_prime = torch.nn.Parameter(torch.rand_like(y))

optimizer = torch.optim.Adam([x_prime], lr=0.001)
print(((x_prime - x) ** 2).sum())
for i in range(10000):
    optimizer.zero_grad()
    loss = wasserstein_distance(x, x_prime).sum()
    if i % 1000 == 0:
        print("loss", loss)
    loss.backward()
    optimizer.step()
