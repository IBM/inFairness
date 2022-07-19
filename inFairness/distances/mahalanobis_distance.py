import torch

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

    def batches_of_sets_of_items(self):
        """ changes the signature of forward such that X1 and X2 are of dimensions B,N,D (batch size, number of items, num features)
        and returns batched pairwise distances B,N,1
        """
        def new_forward(X1, X2):
            x_diff = X1 - X2
            dist = torch.einsum("bnj,ji,bni -> bn",x_diff,self.sigma,x_diff)
            dist = dist.reshape(*(tuple(dist.shape) + (1,)))
            return dist

        self.forward = new_forward

    def forward(self, X1, X2, batches_of_sets_of_items=False):
        """Computes the distance between data samples X1 and X2

        Parameters
        -----------
            X1: torch.Tensor
                Data samples from batch 1 of shape (n_samples, n_features)
            X2: torch.Tensor
                Data samples from batch 2 of shape (n_samples, n_features)
            batches_of_sets_of_items: bool
                if True, X1 and X2 are expected to have (batch_size, n_samples, n_features)
                and the function would return (batch_size, n_samples, 1)

        Returns
        ----------
            dist: torch.Tensor
                Distance between each sample of batch 1 and batch 2.
                Resulting shape is (n_samples, 1). If ``batches_of_sets_of_items`` is True
                it returns pairwise distance tensor with dims (batch_size, n_samples, 1)
        """
        X_diff = X1 - X2

        if batches_of_sets_of_items:
            X_diff = X1 - X2
            dist = torch.einsum("bnj,ji,bni -> bn",X_diff,self.sigma,X_diff)
            dist = dist.reshape(*(tuple(dist.shape) + (1,)))
            return dist
        else:
            dist = torch.sum((X_diff @ self.sigma) * X_diff, dim=-1, keepdim=True)
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
