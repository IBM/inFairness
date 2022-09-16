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


def test_squared_euclidean_dist_state():

    dist = distances.SquaredEuclideanDistance()
    dist.fit(num_dims=5)

    state_dict = dist.state_dict()
    assert "sigma" in state_dict
    assert torch.all(torch.eye(5) == state_dict["sigma"])


def test_protected_euclidean_dist_state():

    protected_attrs = [1]
    num_attrs = 3

    dist = distances.ProtectedEuclideanDistance()
    dist.fit(protected_attrs, num_attrs)

    protected_vec = torch.ones(num_attrs)
    protected_vec[protected_attrs] = 0.0

    state_dict = dist.state_dict()
    assert "protected_vector" in state_dict
    assert torch.all(protected_vec == state_dict["protected_vector"])


def test_svd_distance_state():

    n_features = 50
    n_components = 10

    X_train = torch.rand((100, n_features))

    metric = distances.SVDSensitiveSubspaceDistance()
    metric.fit(X_train, n_components)

    state = metric.state_dict()
    assert "sigma" in state
    sigma = state["sigma"]
    assert sigma.shape == (n_features, n_features)

    metric_new = distances.SVDSensitiveSubspaceDistance()
    metric_new.load_state_dict(state)
    new_state = metric_new.state_dict()
    assert torch.all(new_state["sigma"] == sigma)


def test_explore_distance_state():

    n_features = 50
    n_samples = 100
    
    X1 = torch.rand((n_samples, n_features)).requires_grad_()
    X2 = torch.rand((n_samples, n_features)).requires_grad_()
    Y = torch.randint(low=0, high=2, size=(n_samples,))

    metric = distances.EXPLOREDistance()
    metric.fit(X1, X2, Y, iters=100, batchsize=8)

    state = metric.state_dict()
    assert "sigma" in state
    sigma = state["sigma"]
    assert sigma.shape == (n_features, n_features)

    metric_new = distances.EXPLOREDistance()
    metric_new.load_state_dict(state)
    new_state = metric_new.state_dict()
    assert torch.all(new_state["sigma"] == sigma)


def test_logreg_distance_state():

    n_samples, n_features = 100, 3
    X_train = torch.rand(size=(n_samples, n_features))
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    X_train = (X_train - mean) / std

    protected_attr = torch.randint(low=0, high=2, size=(n_samples, 1))

    X_train[:, 0:1] += protected_attr
    X_train = torch.hstack((X_train, protected_attr))

    metric = distances.LogisticRegSensitiveSubspace()
    metric.fit(X_train, protected_idxs=[3])

    state = metric.state_dict()
    assert "sigma" in state
    sigma = state["sigma"]
    assert sigma.shape == (n_features+1, n_features+1)

    metric_new = distances.EXPLOREDistance()
    metric_new.load_state_dict(state)
    new_state = metric_new.state_dict()
    assert torch.all(new_state["sigma"] == sigma)


def test_wasserstein_dist_state():

    squared_euclidean = distances.SquaredEuclideanDistance()
    squared_euclidean.fit(num_dims=2)
    sigma = squared_euclidean.sigma

    wasserstein_dist = distances.WassersteinDistance()
    wasserstein_dist.fit(sigma)

    state = wasserstein_dist.state_dict()
    assert "sigma" in state
    assert torch.all(state["sigma"] == sigma)

    metric_new = distances.WassersteinDistance()
    metric_new.load_state_dict(state)
    new_state = metric_new.state_dict()
    assert torch.all(new_state["sigma"] == sigma)