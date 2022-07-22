import pytest
import math
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from inFairness import distances


def test_euclidean_distance():

    dist = distances.EuclideanDistance()

    X = torch.FloatTensor([[0.0, 0.0], [1.0, 1.0]])

    Y = torch.FloatTensor([[1.0, 1.0], [1.0, 1.0]])

    res = torch.FloatTensor([[math.sqrt(2)], [0.0]])

    assert torch.all(dist(X, Y) == res)


def test_protected_euclidean_distance():

    protected_attrs = [1]  # Make the second dimension protected attribute
    num_attrs = 3
    dist = distances.ProtectedEuclideanDistance()
    dist.fit(protected_attrs, num_attrs)

    X = torch.FloatTensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )

    Y = torch.FloatTensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    res = torch.FloatTensor(
        [[math.sqrt(2)], [math.sqrt(2)], [math.sqrt(2)], [0.0], [0.0], [0.0]]
    )

    assert torch.all(dist(X, Y) == res), f"{dist(X, Y)} :: {res}"


def test_svd_sensitive_subspace_distance():

    n_samples = 10
    n_features = 50
    n_components = 10

    X_train = torch.rand((100, n_features))

    n_samples = 10
    X1 = torch.rand((n_samples, n_features)).requires_grad_()
    X2 = torch.rand((n_samples, n_features)).requires_grad_()

    metric = distances.SVDSensitiveSubspaceDistance()
    metric.fit(X_train, n_components)

    dist = metric(X1, X2)
    assert list(dist.shape) == [n_samples, 1]
    assert dist.requires_grad == True

    dist = metric(X1, X1)
    assert torch.all(dist == 0)
    assert dist.requires_grad == True


def test_svd_sensitive_subspace_distance_multiple_similar_data():

    n_samples = 10
    n_features = 50
    n_components = 10

    X_train = [torch.rand((100, n_features)) for _ in range(10)]

    n_samples = 10
    X1 = torch.rand((n_samples, n_features)).requires_grad_()
    X2 = torch.rand((n_samples, n_features)).requires_grad_()

    metric = distances.SVDSensitiveSubspaceDistance()
    metric.fit(X_train, n_components)

    dist = metric(X1, X2)
    assert list(dist.shape) == [n_samples, 1]
    assert dist.requires_grad == True

    dist = metric(X1, X1)
    assert torch.all(dist == 0)
    assert dist.requires_grad == True


def test_svd_sensitive_subspace_distance_raises_error():

    n_components = 10
    X_train = None
    metric = distances.SVDSensitiveSubspaceDistance()
    with pytest.raises(TypeError):
        metric.fit(X_train, n_components)


def test_explore_sensitive_subspace_distance():

    n_features = 50

    X1 = torch.rand((100, n_features))
    X2 = torch.rand((100, n_features))
    Y = torch.randint(low=0, high=2, size=(100,))

    n_samples = 10
    X1 = torch.rand((n_samples, n_features)).requires_grad_()
    X2 = torch.rand((n_samples, n_features)).requires_grad_()

    metric = distances.EXPLOREDistance()
    metric.fit(X1, X2, Y, iters=100, batchsize=8)

    dist = metric(X1, X2)
    assert list(dist.shape) == [n_samples, 1]
    assert dist.requires_grad == True

    dist = metric(X1, X1)
    assert torch.all(dist == 0)
    assert dist.requires_grad == True


def test_squared_euclidean_distance():
    x1 = 2 * torch.ones(2)
    x2 = torch.zeros(2)
    dist = distances.SquaredEuclideanDistance()
    dist.fit(num_dims=2)

    distx1x2 = dist(x1, x2)
    assert distx1x2.item() == 8

    distx1x1 = dist(x1, x1)
    assert distx1x1 == 0


def test_logistic_reg_distance_protected_idx():

    X_train = torch.rand(size=(100, 3))
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    X_train = (X_train - mean) / std

    protected_attr = torch.randint(low=0, high=2, size=(100, 1))

    X_train[:, 0:1] += protected_attr
    X_train = torch.hstack((X_train, protected_attr))

    dist = distances.LogisticRegSensitiveSubspace()
    dist.fit(X_train, protected_idxs=[3])

    assert dist.basis_vectors_.shape == (4, 2)
    assert dist.basis_vectors_[0, 0] > dist.basis_vectors_[1, 0]


def test_logistic_reg_distance_no_protected_idx():

    X_train = torch.rand(size=(100, 5))
    protected_attr = torch.randint(low=0, high=2, size=(100, 2)).long()

    dist = distances.LogisticRegSensitiveSubspace()
    dist.fit(X_train, data_SensitiveAttrs=protected_attr)

    assert dist.basis_vectors_.shape == (5, 2)


def test_logistic_reg_distance_raises_error():

    X_train = torch.rand(size=(100, 5))
    protected_attr = torch.randint(low=0, high=2, size=(100, 2)).long()

    dist = distances.LogisticRegSensitiveSubspace()

    with pytest.raises(AssertionError):
        dist.fit(X_train, data_SensitiveAttrs=protected_attr, protected_idxs=[1,2])

    protected_attr = torch.randint(low=0, high=6, size=(100, 2)).long()
    dist = distances.LogisticRegSensitiveSubspace()

    with pytest.raises(AssertionError):
        dist.fit(X_train, protected_attr)


def test_wasserstein_distance():
    """
    uses a SquaredEuclidean special case of a Mahalanobis distance to reduce the set difference between
    2 batches of elements.
    """
    squared_euclidean = distances.SquaredEuclideanDistance()
    wasserstein_dist = distances.BatchedWassersteinDistance(squared_euclidean)
    wasserstein_dist.fit(num_dims=2)

    x1 = torch.randn(3, 10, 2)
    x2 = torch.nn.Parameter(torch.ones_like(x1))
    optimizer = torch.optim.Adam([x2], lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        loss = wasserstein_dist(x1, x2).sum()
        loss.backward()
        optimizer.step()

    """
    if two sets are close in the euclidean space, the sum of the elements in the two sets must add to a similar 
    value
    """
    assert (torch.abs(x1.sum(dim=1).sum(dim=1) - x2.sum(dim=1).sum(dim=1)) < 3.0).all()
