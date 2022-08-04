# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data.sampler import RandomSampler, BatchSampler
# from torch.utils.data import IterableDataset

# from ipdb import set_trace; set_trace()

# def generate_synthetic_LTR_data(majority_proportion = .8, num_queries = 100, num_docs_per_query = 10, seed=0):
#     num_items = num_queries*num_docs_per_query
#     X = np.random.uniform(0,3, size = (num_items,2)).astype(np.float32)
#     relevance = X[:,0] + X[:,1]

#     relevance = np.clip(relevance, 0.0,5.0)
#     majority_status = np.random.choice([True, False], size=num_items, p=[majority_proportion, 1-majority_proportion])
#     X[~majority_status, 1] = 0
#     return [{"Q":X[i], "relevances":relevance[i], "majority_status":majority_status[i]} for i in range(num_items)]


# class QueryIterableDataset(IterableDataset):
#     '''
#     iterable dataset that takes a set of items and indifintely samples sets of such items (queries) per iteration
#     '''
#     def __init__(self, items_dataset, shuffle, query_size):
#         self.dataset = items_dataset
#         self.query_size = query_size
#         self.shuffle = shuffle

#     def __iter__(self):
#         while True:
#             idx = self._infinite_indices()
#             query = [self.dataset[i] for i in next(idx)]
#             query = torch.utils.data.default_collate(query)
#             yield query

#     def _infinite_indices(self):
#         worker_info = torch.utils.data.get_worker_info()
#         seed = 0 if worker_info is None else worker_info.id
#         g = torch.Generator()
#         g.manual_seed(seed)
#         while True:
#             if self.shuffle:
#                 idx = (torch.randperm(len(self.dataset))[:self.query_size]).tolist()
#                 yield idx

# num_docs_per_query = 10
# num_queries = 100
# dataset_train = generate_synthetic_LTR_data(num_queries = num_queries, num_docs_per_query = num_docs_per_query)
# dataloader = torch.utils.data.DataLoader(QueryIterableDataset(dataset_train, True, num_docs_per_query), num_workers=2, batch_size=2)

# # we perform a logistic regression on the dataset to build a sensitive direction
# from sklearn.linear_model import LogisticRegression
# all_data = torch.utils.data.default_collate(dataset_train)
# x = all_data['Q']
# majority_status = all_data['majority_status']
# LR = LogisticRegression(C = 100).fit(x, majority_status)
# sens_directions = torch.tensor(LR.coef_,dtype=torch.float32).T
# print('sensitive directions', sens_directions)


# from inFairness.distances import SensitiveSubspaceDistance, BatchedWassersteinDistance

# distance_q = BatchedWassersteinDistance(SensitiveSubspaceDistance())
# distance_q.fit(sens_directions)

# import torch.nn.functional as F
# class MultilayerPerceptron(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(2, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model1 = MultilayerPerceptron()

# from inFairness.distances import SquaredEuclideanDistance
# distance_y = SquaredEuclideanDistance()
# distance_y.fit(num_dims=num_docs_per_query)


# from inFairness.fairalgo import SenSTIR
# fairalgo1 = SenSTIR(
#     network=model1,
#     distance_q=distance_q,
#     distance_y=distance_y,
#     rho=0.01,
#     eps=0.05,
#     auditor_nsteps=20,
#     auditor_lr=0.05,
#     monte_carlo_samples_ndcg=20,
# )


# class Trainer(object):
#     """Main trainer class that orchestrates the entire learning routine
#     Use this class to start training a model using individual fairness routines

#     Args:
#         dataloader (torch.util.data.DataLoader): training data loader
#         model (inFairness.fairalgo): Individual fairness algorithm
#         optimizer (torch.optim): Model optimizer
#         max_iterations (int): Number of training steps
#     """

#     def __init__(self, dataloader, model, optimizer, max_iterations, print_loss_period=0):

#         self.dataloader = dataloader
#         self.model = model
#         self.optimizer = optimizer
#         self.max_iterations = max_iterations

#         self._dataloader_iter = iter(self.dataloader)
#         self.print_loss_period = print_loss_period

#     def run_step(self):

#         try:
#             data = next(self._dataloader_iter)
#         except StopIteration:
#             self._dataloader_iter = iter(self.dataloader)
#             data = next(self._dataloader_iter)

#         if isinstance(data, list) or isinstance(data, tuple):
#             model_output = self.model(*data)
#         elif isinstance(data, dict):
#             model_output = self.model(**data)
#         else:
#             raise AttributeError(
#                 "Data format not recognized. Only `list`, `tuple`, and `dict` are recognized."
#             )

#         if self.print_loss_period:
#             if self.step_count % self.print_loss_period == 0:
#                 print(f'loss {self.step_count}', model_output.loss)

#         self.optimizer.zero_grad()
#         model_output.loss.backward()

#         self.optimizer.step()

#     def train(self):

#         self.model.train(True)

#         for self.step_count in range(self.max_iterations):
#             self.run_step()


# trainer = Trainer(
#     dataloader=dataloader,
#     model=fairalgo1,
#     optimizer=torch.optim.Adam(fairalgo1.parameters(),lr=0.01),
#     max_iterations = 1000,
#     print_loss_period=40
# )

# trainer.train()

import torch

from inFairness.distances import (
    BatchedWassersteinDistance,
    SensitiveSubspaceDistance,
    SquaredEuclideanDistance,
)
from inFairness.fairalgo import SenSTIR


def generate_test_data(num_batches, queries_per_batch, items_per_query):
    num_features = 2
    item_data = torch.rand(
        num_batches, queries_per_batch, items_per_query, num_features
    )
    relevances = torch.sum(item_data, dim=3)

    # mask the second dimension for some items
    mask = torch.ones(num_batches, queries_per_batch, items_per_query, 1)
    mask = torch.cat([mask, mask.clone().bernoulli_(0.8)], dim=3)
    item_data *= mask

    return item_data, relevances


def test_senstir():
    num_steps = 200
    queries_per_batch = 10
    items_per_query = 5
    feature_size = 2

    # dummy synthetic data
    item_data, relevances = generate_test_data(
        num_steps, queries_per_batch, items_per_query
    )

    # dummy wasserstein distance sensitive on the first dimension
    distance_q = BatchedWassersteinDistance(SensitiveSubspaceDistance())
    distance_q.fit(
        basis_vectors=torch.tensor([[0], [1.0]])
    )  # we use the second dimension in the basis vector because the projection complement will give us the first

    distance_y = SquaredEuclideanDistance()
    distance_y.fit(num_dims=items_per_query)

    # dummy network equally sensitive in both dimensions
    network = torch.nn.Linear(feature_size, 1, bias=None)
    network.weight.data = (
        torch.ones((1, feature_size)) + torch.rand((1, feature_size)) * 0.01
    )

    fair_algo = SenSTIR(
        network,
        distance_q,
        distance_y,
        rho=0.1,
        eps=0.001,
        auditor_nsteps=10,
        auditor_lr=0.05,
        monte_carlo_samples_ndcg=60,
    )
    fair_algo.train()

    optimizer = torch.optim.Adam(fair_algo.parameters(), lr=0.01)

    for i in range(num_steps):
        optimizer.zero_grad()
        loss = fair_algo(item_data[i], relevances[i]).loss
        loss.backward()
        optimizer.step()

    weights = network.weight.data.squeeze()
    # the ratio of the first component of this vector should be greater than 3
    # so that the response of the network should be majorly on the first dimension
    assert weights[0] / weights[1] > 3.0
