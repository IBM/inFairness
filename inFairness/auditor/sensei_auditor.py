import torch
from torch.nn import Parameter

from inFairness.auditor.auditor import Auditor
from inFairness.utils.params import freeze_network, unfreeze_network
from inFairness.utils.datautils import get_device


class SenSeIAuditor(Auditor):
    """SenSeI Auditor implements the functionality to generate worst-case examples
    by solving the following optimization equation:

    .. math:: x_{t_b}' \gets arg\max_{x' \in X}\{d_{Y}(h_{\\theta_t}(X_{t_b}),h_{\\theta_t}(x')) - \lambda_t d_{X}(X_{t_b},x')\}

    Proposed in `SenSeI: Sensitive Set Invariance for Enforcing Individual Fairness <https://arxiv.org/abs/2006.14168>`_

    Parameters
    --------------
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
        distance_y: inFairness.distances.Distance
            Distance metric in the output space
        num_steps: int
            Number of update steps should the auditor perform to find worst-case examples
        lr: float
            Learning rate
    """

    def __init__(
        self, distance_x, distance_y, num_steps, lr, max_noise=0.1, min_noise=-0.1
    ):

        self.distance_x = distance_x
        self.distance_y = distance_y
        self.num_steps = num_steps
        self.lr = lr
        self.max_noise = max_noise
        self.min_noise = min_noise

        super().__init__()

    def generate_worst_case_examples(self, network, x, lambda_param, optimizer=None):
        """Generate worst case example given the input data sample batch `x`

        Parameters
        ------------
            network: torch.nn.Module
                PyTorch network model
            x: torch.Tensor
                Batch of input datapoints
            lambda_param: float
                Lambda weighting parameter as defined in the equation above
            optimizer: torch.optim.Optimizer, optional
                PyTorch Optimizer object. Default: torch.optim.Adam

        Returns
        ---------
            X_worst: torch.Tensor
                Worst case examples for the provided input datapoints
        """

        assert optimizer is None or issubclass(optimizer, torch.optim.Optimizer), (
            "`optimizer` object should either be None or be a PyTorch optimizer "
            + "and an instance of the `torch.optim.Optimizer` class"
        )

        freeze_network(network)
        lambda_param = lambda_param.detach()

        delta = Parameter(
            torch.rand_like(x) * (self.max_noise - self.min_noise) + self.min_noise
        )

        if optimizer is None:
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        else:
            optimizer = optimizer([delta], lr=self.lr)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            x_worst = x + delta
            input_dist = self.distance_x(x, x_worst)
            out_x = network(x)
            out_x_worst = network(x_worst)

            out_dist = self.distance_y(out_x, out_x_worst)

            audit_loss = -(out_dist - lambda_param * input_dist)
            audit_loss.sum().backward()
            optimizer.step()

        unfreeze_network(network)

        return (x + delta).detach()

    def audit(
        self,
        network,
        X_audit,
        Y_audit,
        loss_fn,
        audit_threshold=None,
        lambda_param=None,
        confidence=0.95,
        optimizer=None,
    ):
        """Audit a model for individual fairness

        Parameters
        ------------
            network: torch.nn.Module
                PyTorch network model
            X_audit: torch.Tensor
                Auditing data samples. Shape: (B, *)
            Y_audit: torch.Tensor
                Auditing data samples. Shape: (B)
            loss_fn: torch.nn.Module
                Loss function
            audit_threshold: float, optional
                Auditing threshold to consider a model individually fair or not
                If `audit_threshold` is specified, the `audit` procedure determines
                if the model is individually fair or not.
                If `audit_threshold` is not specified, the `audit` procedure simply
                returns the mean and lower bound of loss ratio, leaving the determination
                of models' fairness to the user.
                Default=None
            lambda_param: float
                Lambda weighting parameter as defined in the equation above
            confidence: float, optional
                Confidence value. Default = 0.95
            optimizer: torch.optim.Optimizer, optional
                PyTorch Optimizer object. Default: torch.optim.SGD

        Returns
        ------------
            audit_response: inFairness.auditor.datainterface.AuditorResponse
                Audit response containing test statistics
        """

        assert optimizer is None or issubclass(optimizer, torch.optim.Optimizer), (
            "`optimizer` object should either be None or be a PyTorch optimizer "
            + "and an instance of the `torch.optim.Optimizer` class"
        )

        device = get_device(X_audit)

        if lambda_param is None:
            lambda_param = torch.tensor(1.0, device=device)
        elif isinstance(lambda_param, float):
            lambda_param = torch.tensor(lambda_param, device=device)

        if optimizer is None:
            optimizer = torch.optim.SGD

        X_worst = self.generate_worst_case_examples(
            network=network, x=X_audit, lambda_param=lambda_param, optimizer=optimizer
        )

        loss_ratio = self.compute_loss_ratio(
            X_audit=X_audit,
            X_worst=X_worst,
            Y_audit=Y_audit,
            network=network,
            loss_fn=loss_fn,
        )

        audit_response = self.compute_audit_result(
            loss_ratio, audit_threshold, confidence
        )

        return audit_response
