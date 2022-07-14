import torch
from functorch import vmap
from geomloss import SamplesLoss

from inFairness.distances import MahalanobisDistances


def batched_and_vectorize(func):
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


def mahalanobis_distance(x, y, sigma):
    """
    computes the mahalanobis distance between 2 vectors of D elements:

    .. math:: MD = (x - y) \\Sigma (x - y)^{'}

    """
    diff = x - y
    return torch.einsum("i,ij,j", diff, sigma, diff)


batched_vectorized_mahalanobis_distance = batched_and_vectorize(mahalanobis_distance)


def mahalanobis_distance_to_batched_wasserstein_distance(
    mahalanobis_dist: MahalanobisDistances,
):
    """
    transforms an Mahalanobis Distance object so that the forward method becomes a differentiable batched
    Wasserstein distance between sets of items. This Wasserstein distance will use the underlying Mahalanobis
    distance as pairwise cost function to solve the optimal transport problem.

    for more information see equation 2.5 of the reference bellow

    References
    ----------
        `Amanda Bower, Hamid Eftekhari, Mikhail Yurochkin, Yuekai Sun:
        Individually Fair Rankings. ICLR 2021`
    """
    mahalanobis_dist.forward = SamplesLoss(
        "sinkhorn",
        blur=0.05,
        cost=lambda x, y: batched_vectorized_mahalanobis_distance(
            x, y, mahalanobis_dist.sigma
        ),
    )
    return mahalanobis_dist
