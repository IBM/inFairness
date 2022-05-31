import torch
from torch import nn

from inFairness.auditor import SenSeIAuditor
from inFairness.fairalgo.datainterfaces import FairModelResponse
from inFairness.utils import datautils


class SenSeI(nn.Module):
    """Implementes the Sensitive Set Invariane (SenSeI) algorithm.

    Proposed in `SenSeI: Sensitive Set Invariance for Enforcing Individual Fairness <https://arxiv.org/abs/2006.14168>`_

    Parameters
    ------------
        network: torch.nn.Module
            Network architecture
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
        distance_y: inFairness.distances.Distance
            Distance metric in the output space
        loss_fn: torch.nn.Module
            Loss function
        rho: float
            :math:`\\rho` parameter in the SenSR algorithm
        eps: float
            :math:`\epsilon` parameter in the SenSR algorithm
        auditor_nsteps: int
            Number of update steps for the auditor to find worst-case examples
        auditor_lr: float
            Learning rate for the auditor
    """

    def __init__(
        self,
        network,
        distance_x,
        distance_y,
        loss_fn,
        rho,
        eps,
        auditor_nsteps,
        auditor_lr,
    ):

        super().__init__()

        self.distance_x = distance_x
        self.distance_y = distance_y
        self.network = network
        self.loss_fn = loss_fn
        self.lamb = None
        self.rho = rho
        self.eps = eps
        self.auditor_nsteps = auditor_nsteps
        self.auditor_lr = auditor_lr

        self.auditor = self.__init_auditor__()

    def __init_auditor__(self):

        auditor = SenSeIAuditor(
            distance_x=self.distance_x,
            distance_y=self.distance_y,
            num_steps=self.auditor_nsteps,
            lr=self.auditor_lr,
        )
        return auditor

    def forward_train(self, X, Y):
        """Forward method during the training phase"""

        device = datautils.get_device(X)
        minlambda = torch.tensor(1e-5, device=device)

        if self.lamb is None:
            self.lamb = torch.tensor(1.0, device=device)
        if type(self.eps) is float:
            self.eps = torch.tensor(self.eps, device=device)

        Y_pred = self.network(X)
        X_worst = self.auditor.generate_worst_case_examples(
            self.network, X, lambda_param=self.lamb
        )
        
        dist_x = self.distance_x(X, X_worst)
        mean_dist_x = dist_x.mean()
        lr_factor = torch.maximum(mean_dist_x, self.eps) / torch.minimum(mean_dist_x, self.eps)

        self.lamb = torch.max(
            torch.stack(
                [minlambda, self.lamb + lr_factor * (mean_dist_x - self.eps)]
            )
        )

        Y_pred_worst = self.network(X_worst)
        fair_loss = torch.mean(
            self.loss_fn(Y_pred, Y) + self.rho * self.distance_y(Y_pred, Y_pred_worst)
        )

        response = FairModelResponse(loss=fair_loss, y_pred=Y_pred)
        return response

    def forward_test(self, X):
        """Forward method during the test phase"""

        Y_pred = self.network(X)
        response = FairModelResponse(y_pred=Y_pred)
        return response

    def forward(self, X, Y=None, *args, **kwargs):
        """Defines the computation performed at every call.

        Parameters
        ------------
            X: torch.Tensor
                Input data
            Y: torch.Tensor
                Expected output data

        Returns
        ----------
            output: torch.Tensor
                Model output
        """

        if self.training:
            return self.forward_train(X, Y)
        else:
            return self.forward_test(X)
