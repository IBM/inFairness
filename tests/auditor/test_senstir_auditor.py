import pytest

import torch
from mock import patch

from inFairness.auditor import SenSTIRAuditor
from inFairness.distances import SensitiveSubspaceDistance, BatchedWassersteinDistance, SquaredEuclideanDistance


def mock_torch_rand_like(*size):
  return torch.ones_like(*size)

@patch("torch.rand_like", mock_torch_rand_like)
def test_sestirauditor_generate_worst_case_examples():
  batch_size = 2
  query_size = 10
  feature_size = 2

  num_steps = 1000
  lr = 0.005
  max_noise = 0.5
  min_noise = -0.5
  lambda_param = torch.tensor(3000.0)

  #let's create a Wasserstein Distance sensitive on the first dimension
  distance_q = BatchedWassersteinDistance(SensitiveSubspaceDistance())
  distance_q.fit(basis_vectors = torch.tensor([[0],[1.]])) #we use the second dimension in the basis vector because the projection complement will give us the first

  #distance between sets of items
  distance_y = SquaredEuclideanDistance()
  distance_y.fit(num_dims=query_size)

  auditor = SenSTIRAuditor(distance_q, distance_y, num_steps, lr, max_noise, min_noise)

  #let's create a dummy network equally sensitive in both dimensions
  network = torch.nn.Linear(feature_size,1, bias=None)
  network.weight.data = torch.ones((1,feature_size))

  #now some dummy batch of queries
  Q = torch.randn(batch_size,query_size,feature_size)

  Q_worst = auditor.generate_worst_case_examples(network, Q, lambda_param, torch.optim.Adam)

  #since the first dimension is sensitive, the examples should differ quite a bit in the second dimension while being similar in the first
  first_dim_Q = Q[:,:,0]
  second_dim_Q = Q[:,:,1]

  first_dim_Q_worst = Q_worst[:,:,0]
  second_dim_Q_worst = Q_worst[:,:,1]

  # if two sets differ, their values should add to a high value
  assert (torch.abs(second_dim_Q.sum(1) - second_dim_Q_worst.sum(1)) > 10.0).all()

  # if two sets are close, their sum should add to a similar value
  assert (torch.abs(first_dim_Q.sum(1) - first_dim_Q_worst.sum(1)) < 1.0).all()

  auditor.compute_loss_ratio(Q,Q_worst,)

if __name__ == "__main__":
  from ipdb import set_trace; set_trace()
  test_sestirauditor_generate_worst_case_examples()
