from re import X
import pytest
import numpy as np

from inFairness.auditor import Auditor
from mock import patch
import torch
from torch.nn import functional as F


def mock_adam_optim(
    params, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
):
    return torch.optim.SGD(params, lr=lr)


def my_dist(s, t):
    return torch.norm(s - t, dim=0).pow(2)


class MockPerceptron(torch.nn.Module):
    def __init__(self, xdim, ydim):
        super().__init__()
        self.fc = torch.nn.Linear(xdim, ydim, dtype=float, bias=False)

    def forward(self, x):
        output = self.fc(x)
        return output


def mock_torch_rand(*size):
    return torch.ones(*size)


def test_auditor_loss_ratio():

    xdim = 50
    ydim = 1
    B = 100

    network = MockPerceptron(xdim, ydim)
    loss_fn = F.l1_loss

    auditor = Auditor()

    X_audit = torch.rand(size=(B, xdim), dtype=torch.float64)
    X_worst = torch.rand(size=(B, xdim), dtype=torch.float64)
    Y_audit = torch.rand(size=(B, ydim), dtype=torch.float64)

    loss_ratio = auditor.compute_loss_ratio(X_audit, X_worst, Y_audit, network, loss_fn)

    assert np.array_equal(loss_ratio.shape, [B, 1])
