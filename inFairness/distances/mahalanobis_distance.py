import torch
import numpy as np
from functorch import vmap

from inFairness.distances.distance import Distance


class MahalanobisDistances(Distance):
    """Base class implementing the Generalized Mahalanobis Distances

    Mahalanobis distance between two points X1 and X2 is computed as:

    .. math:: \\text{dist}(X_1, X_2) = (X_1 - X_2) \\Sigma (X_1 - X_2)^{T}
    """

    def __init__(self):
        super().__init__()

        self.sigma = None
        self.device = torch.device("cpu")

    def to(self, device):
        """Moves distance metric to a particular device

        Parameters
        ------------
            device: torch.device
        """

        assert (
            self.sigma is not None
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.sigma = self.sigma.to(self.device)

    def fit(self, sigma):
        """Fit Mahalanobis Distance metric

        Parameters
        ------------
            sigma: torch.Tensor
                    Covariance matrix
        """

        self.sigma = sigma

    @staticmethod
    def __compute_dist__(X1, X2, sigma):
        """Computes the distance between two data samples x1 and x2

        Parameters
        -----------
            X1: torch.Tensor
                Data sample of shape (n_features) or (N, n_features)
            X2: torch.Tensor
                Data sample of shape (n_features) or (N, n_features)

        Returns:
            dist: torch.Tensor
                Distance between points x1 and x2. Shape: (N)
        """

        # unsqueeze batch dimension if a vector is passed
        if len(X1.shape) == 1:
            X1 = X1.unsqueeze(0)
        if len(X2.shape) == 1:
            X2 = X2.unsqueeze(0)

        X_diff = X1 - X2
        dist = torch.sum((X_diff @ sigma) * X_diff, dim=-1, keepdim=True)
        return dist

    def forward(self, X1, X2, itemwise_dist=True):
        """Computes the distance between data samples X1 and X2

        Parameters
        -----------
            X1: torch.Tensor
                Data samples from batch 1 of shape (n_samples_1, n_features)
            X2: torch.Tensor
                Data samples from batch 2 of shape (n_samples_2, n_features)
            itemwise_dist: bool, default: True
                Compute the distance in an itemwise manner or pairwise manner.

                In the itemwise fashion (`itemwise_dist=False`), distance is
                computed between the ith data sample in X1 to the ith data sample
                in X2. Thus, the two data samples X1 and X2 should be of the same shape

                In the pairwise fashion (`itemwise_dist=False`), distance is
                computed between all the samples in X1 and all the samples in X2.
                In this case, the two data samples X1 and X2 can be of different shapes.

        Returns
        ----------
            dist: torch.Tensor
                Distance between samples of batch 1 and batch 2.

                If `itemwise_dist=True`, item-wise distance is returned of
                shape (n_samples, 1)

                If `itemwise_dist=False`, pair-wise distance is returned of
                shape (n_samples_1, n_samples_2)
        """

        if itemwise_dist:
            np.testing.assert_array_equal(
                X1.shape,
                X2.shape,
                err_msg="X1 and X2 should be of the same shape for itemwise distance computation",
            )
            dist = self.__compute_dist__(X1, X2, self.sigma)
        else:
            X1 = X1.unsqueeze(0) if len(X1.shape) == 2 else X1  # (B, N, D)
            X2 = X2.unsqueeze(0) if len(X2.shape) == 2 else X2  # (B, M, D)

            nsamples_x1 = X1.shape[1]
            nsamples_x2 = X2.shape[1]
            dist_shape = (-1, nsamples_x1, nsamples_x2)

            vdist = vmap(
                vmap(
                    vmap(self.__compute_dist__, in_dims=(None, 0, None)),
                    in_dims=(0, None, None),
                ),
                in_dims=(0, 0, None),
            )

            dist = vdist(X1, X2, self.sigma).view(dist_shape)

        return dist


class SquaredEuclideanDistance(MahalanobisDistances):
    """
    computes the squared euclidean distance as a special case of the mahalanobis distance where:

    .. math:: \\Sigma= I_{num_dims}
    """

    def __init__(self):
        super().__init__()
        self.num_dims_ = None

    def fit(self, num_dims: int):
        """Fit Square Euclidean Distance metric

        Parameters
        -----------------
            num_dims: int
                the number of dimensions of the space in which the Squared Euclidean distance will be used.
        """

        self.num_dims_ = num_dims
        sigma = torch.eye(self.num_dims_).detach()
        super().fit(sigma)
