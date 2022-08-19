import torch
from functorch import vmap

from inFairness.utils.plackett_luce import PlackettLuce
from inFairness.utils.misc import vect_gather


def discounted_cumulative_gain(relevances):
    numerator = torch.pow(torch.tensor([2.0]), relevances)
    denominator = torch.log2(torch.arange(len(relevances), dtype=torch.float) + 2)
    return (numerator / denominator).sum()


def normalized_discounted_cumulative_gain(relevances):
    """takes a vector of relevances and computes the normalized discounted cumulative gain
    taken from (wikipedia)[https://en.wikipedia.org/wiki/Discounted_cumulative_gain]

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


def log_expected_ndcg(montecarlo_samples, scores, relevances):
    """
    uses monte carlo samples to estimate the expected normalized discounted cumulative reward
    by using REINFORCE. See section 2 of the reference bellow.

    Parameters
    -------------
    scores: torch.Tensor of dimension B,N
      predicted scores for the objects in a batch of queries

    relevances: torch.Tensor of dimension B,N
      corresponding true relevances of such objects

    Returns
    ------------
    expected_ndcg: torch.Tensor of dimension B
      monte carlo approximation of the expected ndcg by sampling from a Plackett-Luce
      distribution parameterized by :param:`scores`

    References
    ----------
        `Amanda Bower, Hamid Eftekhari, Mikhail Yurochkin, Yuekai Sun:
        Individually Fair Rankings. ICLR 2021`
    """
    prob_dist = PlackettLuce(scores)
    mc_rankings = prob_dist.sample((montecarlo_samples,))
    mc_log_prob = prob_dist.log_prob(mc_rankings)

    mc_relevances = vect_gather(relevances, 1, mc_rankings)
    mc_ndcg = monte_carlo_vect_ndcg(mc_relevances)

    expected_utility = (mc_ndcg * mc_log_prob).mean(dim=0)
    return expected_utility
