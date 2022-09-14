import pytest
import torch

from inFairness import distances


def test_mahalanobis_dist_state_buffer_set():

    dist = distances.MahalanobisDistances()
    sigma = torch.rand(size=(10, 10))
    dist.fit(sigma)

    state_dict = dist.state_dict()
    assert "sigma" in state_dict
    assert torch.all(state_dict["sigma"] == sigma)

    sigma = torch.rand(size=(10, 10))
    dist.fit(sigma)

    state_dict = dist.state_dict()
    assert "sigma" in state_dict
    assert torch.all(state_dict["sigma"] == sigma)


def test_mahalanobis_dist_state_update():

    dist = distances.MahalanobisDistances()
    sigma = torch.rand(size=(10, 10))
    dist.fit(sigma)

    state_dict = dist.state_dict()
    assert  "sigma" in state_dict
    assert torch.all(state_dict["sigma"] == sigma)

    dist1 = distances.MahalanobisDistances()
    dist1.load_state_dict(state_dict)
    
    state_dict1 = dist1.state_dict()
    assert "sigma" in state_dict1
    assert torch.all(state_dict1["sigma"] == sigma)
