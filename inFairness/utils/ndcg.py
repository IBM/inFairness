import torch
from functorch import vmap


def discounted_cumulative_gain(relevances):
    numerator = torch.pow(torch.tensor([2.0]), relevances)
    denominator = torch.log2(torch.arange(len(relevances), dtype=torch.float) + 2)
    return (numerator / denominator).sum()


def normalized_discounted_cumulative_gain(relevances):
    """Takes a vector of relevances and computes the normalized discounted cumulative gain
    Refer (Wikipedia - Discounted Cumulative Gain)[https://en.wikipedia.org/wiki/Discounted_cumulative_gain]
    for more information.

    Parameters
    ---------
      relevances: torch.Tensor
        vector of dimension N where each element is the relevance of some objects in a particular order

    Returns
    -------
      normalized_discounted_cumulative_gain: torch.Tensor
        scalar value corresponding to the normalized discounted cumulative gain
    """
    dcg = discounted_cumulative_gain(relevances)
    sorted_rels, _ = torch.sort(relevances, descending=True)
    idcg = discounted_cumulative_gain(sorted_rels)
    return dcg / idcg


"""
vectorizes :func: `normalized_discounted_cumulative_gain` to work on a batch of vectors of relevances
given in a tensor of dimensions B,N. The output would be the NDCG on the last dimmension. And it's batched
version would return B samples.
"""
vect_normalized_discounted_cumulative_gain = vmap(
    normalized_discounted_cumulative_gain, in_dims=(0)
)

"""
Adds a further outer dimension to the vectorized normalized discounted cumulative gain so it works 
on monte carlo samples of rankings (e.g. samples of a plackett-luce distribution).

This function would take a tensor of size S,B,N and return a tensor of size S,B with the
ndcg of each vector.
"""
monte_carlo_vect_ndcg = vmap(vect_normalized_discounted_cumulative_gain, in_dims=(0,))
