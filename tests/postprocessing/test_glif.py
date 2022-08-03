import pytest
import torch
import torch.nn.functional as F
import numpy as np

from inFairness.distances import EuclideanDistance
from inFairness.postprocessing import GraphLaplacianIF


def test_postprocess_incorrectargs():

    params = (1.0, 1.0, 100.0, True)
    dist_x = EuclideanDistance()

    pp = GraphLaplacianIF(dist_x)

    with pytest.raises(AssertionError):
        pp.postprocess(None, *params)

    with pytest.raises(AssertionError):
        pp.postprocess("coordinate-descent", *params)


@pytest.mark.parametrize(
    "lambda_param,scale,threshold,normalize,dim",
    [
        (1.0, 1.0, 100.0, True, 2),
        (1.0, 1.0, 100.0, False, 2),
        (1.0, 1.0, 100.0, True, 10),
        (1.0, 1.0, 100.0, False, 10),
    ],
)
def test_postprocess_exact(lambda_param, scale, threshold, normalize, dim):

    B, E = 50, 100
    X = torch.rand(size=(B, E))
    Y = torch.rand(size=(B, dim))
    Y = F.softmax(Y, dim=-1)

    dist_x = EuclideanDistance()
    
    pp = GraphLaplacianIF(dist_x)
    pp.add_datapoints(X, Y)

    y_pp = pp.postprocess("exact", lambda_param, scale, threshold, normalize)

    assert np.array_equal(list(Y.shape), list(y_pp.shape))


@pytest.mark.parametrize(
    "lambda_param,scale,threshold,normalize,dim",
    [
        (1.0, 1.0, 100.0, True, 2),
        (1.0, 1.0, 100.0, False, 2),
        (1.0, 1.0, 100.0, True, 10),
        (1.0, 1.0, 100.0, False, 10),
    ],
)
def test_postprocess_coo(lambda_param, scale, threshold, normalize, dim):

    B, E = 50, 100
    X = torch.rand(size=(B, E))
    Y = torch.rand(size=(B, dim))
    Y = F.softmax(Y, dim=-1)

    dist_x = EuclideanDistance()
    
    pp = GraphLaplacianIF(dist_x)
    pp.add_datapoints(X, Y)

    y_pp = pp.postprocess(
        "coordinate-descent", lambda_param, scale, 
        threshold, normalize, batchsize=6, epochs=20
    )

    assert np.array_equal(list(Y.shape), list(y_pp.shape))
