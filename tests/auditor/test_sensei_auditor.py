import pytest
import numpy as np
from mock import patch

import torch
from torch.nn import functional as F

from inFairness.auditor import SenSeIAuditor


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


def test_sensei_init():
    xdim = 3
    ydim = 1

    n_fair_steps = 1
    fair_lr = 1.0
    network = MockPerceptron(xdim, ydim)
    lambda_param = torch.tensor(1.0)
    distance_x = my_dist
    distance_y = my_dist

    n_fair_steps = 100
    fair_lr = 100

    sensei = SenSeIAuditor(
        distance_x=distance_x, distance_y=distance_y, num_steps=n_fair_steps, lr=fair_lr
    )

    assert sensei.num_steps == n_fair_steps
    assert sensei.lr == fair_lr


@patch("torch.optim.Adam", mock_adam_optim)
@patch("torch.rand", mock_torch_rand)
def test_sensrauditor_generate_worse_case_examples():
    minibatch_size = 2
    xdim = 3
    ydim = 1

    n_fair_steps = 1
    fair_lr = 1.0
    max_noise = 0.2
    min_noise = 0.0
    x = torch.from_numpy(np.ones([minibatch_size, xdim]))
    y = torch.from_numpy(np.zeros([minibatch_size, ydim]))
    network = MockPerceptron(xdim, ydim)
    lamb = torch.tensor(1.0)
    distance_x = my_dist
    distance_y = my_dist

    for param in network.parameters():
        param.data.fill_(float(1.0))

    se_auditor = SenSeIAuditor(
        distance_x=distance_x,
        distance_y=distance_y,
        num_steps=n_fair_steps,
        lr=fair_lr,
        max_noise=max_noise,
        min_noise=min_noise,
    )

    output = se_auditor.generate_worst_case_examples(
        network=network, x=x, lambda_param=lamb
    )

    assert np.array_equal(list(output.size()), list(x.size()))


@pytest.mark.parametrize(
    "audit_threshold,lambda_param,confidence,optimizer",
    [
        (None, None, 0.95, None),
        (None, None, 0.95, torch.optim.Adam),
        (1.25, None, 0.95, None),
        (1.25, 0.25, 0.85, torch.optim.Adam),
    ],
)
def test_sensei_auditing(audit_threshold, lambda_param, confidence, optimizer):

    xdim = 50
    ydim = 1
    B = 100

    network = MockPerceptron(xdim, ydim)
    loss_fn = F.mse_loss
    distance_x = my_dist
    distance_y = my_dist
    n_fair_steps = 10
    fair_lr = 0.01

    auditor = SenSeIAuditor(
        distance_x=distance_x, distance_y=distance_y, num_steps=n_fair_steps, lr=fair_lr
    )

    X_audit = torch.rand(size=(B, xdim), dtype=torch.float64)
    Y_audit = torch.rand(size=(B, ydim), dtype=torch.float64)

    response = auditor.audit(
        network,
        X_audit,
        Y_audit,
        loss_fn,
        audit_threshold,
        lambda_param,
        confidence,
        optimizer,
    )

    assert response.lossratio_mean is not None and isinstance(
        response.lossratio_mean, float
    )
    assert response.lossratio_std is not None and isinstance(
        response.lossratio_std, float
    )
    assert response.lower_bound is not None and isinstance(response.lower_bound, float)

    if audit_threshold is None:
        assert response.threshold is None
        assert response.pval is None
        assert response.confidence is None
        assert response.is_model_fair is None
    else:
        assert response.threshold is not None and isinstance(response.threshold, float)
        assert response.pval is not None and isinstance(response.pval, float)
        assert response.confidence == confidence
        assert response.is_model_fair is not None and isinstance(
            response.is_model_fair, bool
        )
