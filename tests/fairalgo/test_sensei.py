import pytest
import numpy as np

from inFairness.auditor import SenSeIAuditor
from inFairness.fairalgo import SenSeI
from mock import patch
import torch
from torch.nn import functional as F


def mock_generate_worst_case_examples(cls, network, x, lambda_param):
    return torch.ones_like(x) * -1.0


def mock_dist(s, t):
    return torch.norm(s - t, dim=0).pow(2)


class MockPerceptron(torch.nn.Module):
    def __init__(self, xdim, ydim):
        super().__init__()
        self.fc = torch.nn.Linear(xdim, ydim, dtype=float, bias=False)

    def forward(self, x):
        output = self.fc(x)
        return output


@patch(
    "inFairness.auditor.SenSeIAuditor.generate_worst_case_examples",
    mock_generate_worst_case_examples,
)
def test_sensei_forward_train():
    minibatch_size = 2
    xdim = 3
    ydim = 1

    n_fair_steps = 1
    lr = 1.0
    max_noise = 0.2
    min_noise = 0.0
    x = torch.from_numpy(np.ones([minibatch_size, xdim]))
    y = torch.from_numpy(np.zeros([minibatch_size, ydim]))
    network = MockPerceptron(xdim, ydim)
    loss_fn = F.mse_loss
    rho = 1.0
    eps = 1.0
    distance_x = mock_dist
    distance_y = mock_dist

    for param in network.parameters():
        param.data.fill_(float(1.0))

    sensei = SenSeI(
        network, distance_x, distance_y, loss_fn, rho, eps, n_fair_steps, lr
    )
    response = sensei.forward(x, y)

    assert torch.abs(torch.mean(response.loss) - torch.tensor(81.0)) < 0.000001
    assert torch.abs(torch.mean(response.y_pred) - torch.tensor(3.0)) < 0.000001
    assert isinstance(sensei.auditor, SenSeIAuditor)
