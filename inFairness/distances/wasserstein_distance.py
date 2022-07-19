import torch
from functorch import vmap
from geomloss import SamplesLoss

from inFairness.distances import MahalanobisDistances, Distance


class BatchedWassersteinDistance(Distance):
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

    def __init__(self, distance: Distance):
        super().__init__()
        assert isinstance(
            distance, MahalanobisDistances
        ), "only MahalanobisDistances are supported"
        self.distance = distance
        self.batch_cost_function = self.batch_and_vectorize(self.mahalanobis_distance)

    def forward(self, x, y):
        batched_wasserstein_distance_loss = SamplesLoss(
            "sinkhorn",
            blur=0.05,
            cost=lambda x, y: self.batch_cost_function(x, y, self.distance.sigma),
        )
        return batched_wasserstein_distance_loss(x, y)

    def fit(self, *args, **kwargs):
        self.distance.fit(*args, **kwargs)

    @staticmethod
    def mahalanobis_distance(x, y, sigma):
        """
        computes the mahalanobis distance between 2 vectors of D elements:

        .. math:: MD = (x - y) \\Sigma (x - y)^{'}

        """
        diff = x - y
        return torch.einsum("i,ij,j", diff, sigma, diff)

    @staticmethod
    def batch_and_vectorize(func):
        """
        takes a function with 3 arguments x,y,p and vectorizes it such that the resulting function takes a batch of
        sets of items for both x and y (with dimmensions B, N, D and B, M, D where B and D are the batch and feature sizes and N and M are the number of items on each batch)
        and applies the function to the outer product in the items dimension to return a matrix C with dimmensions B,N,M.

        In the particular case where func is a distance, C is the distance between all possible pairs of items on each batch
        from X and Y.
        """
        vect1 = vmap(func, in_dims=(None, 0, None))
        vect2 = vmap(vect1, in_dims=(0, None, None))
        batched_vectorized_function = vmap(vect2, in_dims=(0, 0, None))
        return batched_vectorized_function
