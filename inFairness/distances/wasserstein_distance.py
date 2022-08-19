import torch
from functorch import vmap
from ot import emd2

from inFairness.distances import MahalanobisDistances, Distance


class BatchedWassersteinDistance(MahalanobisDistances):
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

    def __init__(self, distance: MahalanobisDistances):
        super().__init__()
        assert isinstance(
            distance, MahalanobisDistances
        ), "only MahalanobisDistances are supported"
        self.distance = distance

    def forward(self, x, y):
        """computes a batch wasserstein distance implied by the cost function represented by an
        underlying mahalanobis distance.

        Parameters
        ---------
        x,y: torch.Tensor
            should be of dimensions B,N,D and B,M,D

        Returns
        --------
        batched_wassenstein_distance: torch.Tensor
            dimension B
        """
        costs = self.distance(x, y, itemwise_dist=False)
        uniformx = torch.ones(x.shape[1]) / x.shape[1]
        uniformy = torch.ones(y.shape[1]) / y.shape[1]
        num_batches = x.shape[0]
        batched_wasserstein_distance_loss = torch.stack(
            [emd2(uniformx, uniformy, costs[j]) for j in range(num_batches)]
        )
        return batched_wasserstein_distance_loss

    def fit(self, *args, **kwargs):
        self.distance.fit(*args, **kwargs)
