import torch
from torch import nn

from inFairness.auditor import SenSRAuditor
from inFairness.fairalgo.datainterfaces import FairModelResponse
from inFairness.utils import datautils


class SenSR(nn.Module):
    """Implementes the Sensitive Subspace Robustness (SenSR) algorithm.

    Proposed in `Training individually fair ML models with sensitive subspace robustness <https://arxiv.org/abs/1907.00020>`_

    Parameters
    ------------
        network: torch.nn.Module
            Network architecture
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
        loss_fn: torch.nn.Module
            Loss function
        eps: float
            :math:`\epsilon` parameter in the SenSR algorithm
        lr_lamb: float
            :math:`\lambda` parameter in the SenSR algorithm
        lr_param: float
            :math:`\\alpha` parameter in the SenSR algorithm
        auditor_nsteps: int
            Number of update steps for the auditor to find worst-case examples
        auditor_lr: float
            Learning rate for the auditor
    """

    def __init__(
        self,
        network,
        distance_x,
        loss_fn,
        eps,
        lr_lamb,
        lr_param,
        auditor_nsteps,
        auditor_lr,
    ):

        super().__init__()

        self.distance_x = distance_x
        self.network = network
        self.loss_fn = loss_fn
        self.lambda_param = None
        self.eps = eps
        self.lr_lamb = lr_lamb
        self.lr_param = lr_param
        self.auditor_nsteps = auditor_nsteps
        self.auditor_lr = auditor_lr

        self.auditor = self.__init_auditor__()

    def __init_auditor__(self):

        auditor = SenSRAuditor(
            loss_fn=self.loss_fn,
            distance_x=self.distance_x,
            num_steps=self.auditor_nsteps,
            lr=self.auditor_lr,
        )
        return auditor

    def forward_train(self, X, Y):
        """Forward method during the training phase"""

        device = datautils.get_device(X)

        if self.lambda_param is None:
            self.lambda_param = torch.tensor(1.0, device=device)

        Y_pred = self.network(X)
        X_worst = self.auditor.generate_worst_case_examples(
            self.network, X, Y, lambda_param=self.lambda_param
        )

        self.lambda_param = torch.max(
            torch.stack(
                [
                    torch.tensor(0.0, device=device),
                    self.lambda_param
                    - self.lr_lamb * (self.eps - self.distance_x(X, X_worst).mean()),
                ]
            )
        )

        Y_pred_worst = self.network(X_worst)
        fair_loss = torch.mean(self.lr_param * self.loss_fn(Y_pred_worst, Y))

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
