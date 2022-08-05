import torch
from geomloss import SamplesLoss

from inFairness.distances import MahalanobisDistances, Distance


class WassersteinDistance(MahalanobisDistances):
    """computes a batched Wasserstein Distance for pairs of sets of items on each batch in the tensors
    with dimensions B, N, D and B, M, D where B and D are the batch and feature sizes and N and M are the number of items on each batch.

    Currently only supporting distances inheriting from :class: `MahalanobisDistances`.

    transforms an Mahalanobis Distance object so that the forward method becomes a differentiable batched
    Wasserstein distance between sets of items. This Wasserstein distance will use the underlying Mahalanobis
    distance as pairwise cost function to solve the optimal transport problem.

    for more information see equation 2.5 of the reference bellow

    References
    ----------
        `Amanda Bower, Hamid Eftekhari, Mikhail Yurochkin, Yuekai Sun:
        Individually Fair Rankings. ICLR 2021`
    """

    def __init__(self):
        super().__init__()

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        """computes a batch wasserstein distance implied by the cost function represented by an
        underlying mahalanobis distance.

        Parameters
        --------------
        X1: torch.Tensor
            Data sample of shape (B, N, D)
        X2: torch.Tensor
            Data sample of shape (B, M, D)

        Returns
        --------
        dist: torch.Tensor
            Wasserstein distance of shape (B) between batch samples in X1 and X2
        """
        
        wasserstein_distance_loss = SamplesLoss(
            "sinkhorn",
            blur=0.05,
            cost=lambda x, y: super().forward(x, y, itemwise_dist=False),
        )
        dist = WassersteinDistance(X1, X2)

        return dist
