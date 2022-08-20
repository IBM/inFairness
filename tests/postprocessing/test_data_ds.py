import pytest
import torch
import numpy as np

from inFairness.distances import EuclideanDistance
from inFairness.postprocessing.data_ds import PostProcessingDataStore


def test_add_data():

    ntries = 10
    B, D = 10, 50
    distance_x = EuclideanDistance()
    data_ds = PostProcessingDataStore(distance_x)
    counter = 0

    for _ in range(ntries):
        X = torch.rand(size=(B, D))
        Y = torch.rand(size=(B,))

        counter += B

        data_ds.add_datapoints(X, Y)
        assert data_ds.n_samples == counter
        assert np.array_equal(
            list(data_ds.distance_matrix.shape),
            [counter, counter]
        )


def test_reset_data():

    B, D = 10, 50
    distance_x = EuclideanDistance()
    data_ds = PostProcessingDataStore(distance_x)

    X = torch.rand(size=(B, D))
    Y = torch.rand(size=(B,))

    data_ds.add_datapoints(X, Y)

    assert data_ds.n_samples == B
    assert np.array_equal(list(data_ds.distance_matrix.shape), [B, B])

    data_ds.reset()
    assert data_ds.n_samples == 0
    assert data_ds.distance_matrix is None
    assert data_ds.data_X is None
    assert data_ds.data_Y is None
