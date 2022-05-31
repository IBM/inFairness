import pytest
import numpy as np

from inFairness.utils import datautils
from inFairness.utils.datautils import include_exclude_terms


def test_datapair_generation_1data_random():

    # Generate data pairs fewer than possible
    data = np.random.random(size=(100, 5, 5))
    npairs = 10
    pair_idxs = datautils.generate_data_pairs(n_pairs=npairs, datasamples_1=data)
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs same as possible
    data = np.random.random(size=(10,))
    npairs = 100
    pair_idxs = datautils.generate_data_pairs(n_pairs=npairs, datasamples_1=data)
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs more as possible. should raise error
    data = np.random.random(size=(10,))
    npairs = 101

    with pytest.raises(Exception):
        pair_idxs = datautils.generate_data_pairs(n_pairs=npairs, datasamples_1=data)


def test_datapair_generation_2data_random():

    # Generate data pairs fewer than possible
    data1 = np.random.random(size=(100, 5, 5))
    data2 = np.random.random(size=(200, 3))
    npairs = 10

    pair_idxs = datautils.generate_data_pairs(
        n_pairs=npairs, datasamples_1=data1, datasamples_2=data2
    )
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs same as total possible
    data1 = np.random.random(size=(10,))
    data2 = np.random.random(size=(20, 1, 4))
    npairs = 200
    pair_idxs = datautils.generate_data_pairs(
        n_pairs=npairs, datasamples_1=data1, datasamples_2=data2
    )
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs more as possible. should raise error
    data1 = np.random.random(size=(10, 6, 2))
    data2 = np.random.random(size=(5, 2))
    npairs = 51

    with pytest.raises(Exception):
        pair_idxs = datautils.generate_data_pairs(
            n_pairs=npairs, datasamples_1=data1, datasamples_2=data2
        )


def test_datapair_generation_1data_comparator():

    # Generate data pairs fewer than possible
    data = np.random.random(size=(100, 5, 5))
    npairs = 10
    comparator = lambda x, y: np.array_equal(x, y)

    pair_idxs = datautils.generate_data_pairs(
        n_pairs=npairs, datasamples_1=data, comparator=comparator
    )
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs more as possible. should raise error
    data = np.random.random(size=(10,))
    npairs = 11
    comparator = lambda x, y: np.array_equal(x, y)

    with pytest.raises(Exception):
        pair_idxs = datautils.generate_data_pairs(
            n_pairs=npairs, datasamples_1=data, comparator=comparator
        )


def test_datapair_generation_2data_comparator():

    # Generate data pairs fewer than possible
    data1 = np.random.random(size=(100, 5, 5))
    data2 = np.random.random(size=(50, 5, 5))
    npairs = 10
    comparator = lambda x, y: not np.array_equal(x, y)

    pair_idxs = datautils.generate_data_pairs(
        n_pairs=npairs, datasamples_1=data1, datasamples_2=data2, comparator=comparator
    )
    assert pair_idxs.shape == (npairs, 2)

    # Generate data pairs more as possible. should raise error
    data1 = np.random.random(size=(10, 5, 5))
    data2 = data1 + 1.0
    npairs = 1
    comparator = lambda x, y: np.array_equal(x, y)

    with pytest.raises(Exception):
        pair_idxs = datautils.generate_data_pairs(
            n_pairs=npairs,
            datasamples_1=data1,
            datasamples_2=data2,
            comparator=comparator,
        )


def test_include_exclude_terms():
    data_terms = ["a", "c", "b"]
    terms = include_exclude_terms(data_terms, include=["b", "c"])
    assert terms == ["b", "c"]

    terms = include_exclude_terms(data_terms, exclude=["a"])
    assert terms == ["b", "c"]

    terms = include_exclude_terms(data_terms)
    assert terms == ["a", "b", "c"]
