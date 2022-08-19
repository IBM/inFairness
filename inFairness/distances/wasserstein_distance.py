import torch
from ot import emd2

from inFairness.distances import MahalanobisDistances


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

        costs = super().forward(X1, X2, itemwise_dist=False)
        uniform_x1 = torch.ones(X1.shape[1]) / X1.shape[1]
        uniform_x2 = torch.ones(X2.shape[1]) / X2.shape[1]
        num_batches = X1.shape[0]

        dist = torch.stack(
            [emd2(uniform_x1, uniform_x2, costs[j]) for j in range(num_batches)]
        )
        return dist
