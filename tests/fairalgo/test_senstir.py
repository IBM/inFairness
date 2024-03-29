import torch

from inFairness.distances import (
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

    # let's create a Sensitive subspace distance in the input space
    distance_x = SensitiveSubspaceDistance()
    # we use the second dimension in the basis vector because the projection complement will give us the first
    basis_vectors_ = torch.tensor([[0], [1.]])
    distance_x.fit(basis_vectors_)

    distance_y = SquaredEuclideanDistance()
    distance_y.fit(num_dims=items_per_query)

    # dummy network equally sensitive in both dimensions
    network = torch.nn.Linear(feature_size, 1, bias=None)
    network.weight.data = (
        torch.ones((1, feature_size)) + torch.rand((1, feature_size)) * 0.01
    )

    fair_algo = SenSTIR(
        network,
        distance_x,
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
