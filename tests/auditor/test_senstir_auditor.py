import pytest

import torch

from inFairness.auditor import SenSTIRAuditor
from inFairness.distances import SensitiveSubspaceDistance, BatchedWassersteinDistance, SquaredEuclideanDistance

def test_sestirauditor_generate_worst_case_examples():
  batch_size = 2
  query_size = 10
  feature_size = 2

  num_steps = 1000
  lr = 0.01
  max_noise = 0.5
  min_noise = -0.5
  
  data = torch.randn(batch_size,query_size,feature_size)

  #let's create a Wasserstein Distance sensitive on the second dimension
  distance_q = BatchedWassersteinDistance(SensitiveSubspaceDistance())
  distance_q.fit(basis_vectors = torch.tensor([[0],[1.]]))

  #batched pairwise distance in the output space
  distance_y = SquaredEuclideanDistance()
  distance_y.fit(num_dims=feature_size)

  #let's create a dummy network equally sensitive in both dimensions
  network = torch.nn.Linear(feature_size,1, bias=None)
  network.weight.data = torch.ones((1,feature_size))






  

