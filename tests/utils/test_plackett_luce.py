import torch
from torch.nn.parameter import Parameter
from functorch import vmap

from inFairness.utils import plackett_luce
from inFairness.utils.plackett_luce import PlackettLuce
from inFairness.utils.ndcg import vect_normalized_discounted_cumulative_gain as v_ndcg


vect_gather = vmap(torch.gather, in_dims=(None,None, 0))
batched_v_ndcg = vmap(v_ndcg, in_dims=(0))


def test_batch_plackett_luce():
  """
  the idea of this test is to use normalized discounted cumulative gain to evaluate how
  good the underlying plackett_luce distribution approximates some ideal relevance

  after optimization, the parameterized dummy_logits should assign the highest value to
  the most relevant item in the query.
  """

  relevances1 = torch.arange(3,dtype=torch.float)
  relevances2 = torch.arange(2,-1,-1, dtype=torch.float)
  relevances = torch.stack([relevances1, relevances2])

  montecarlo_samples = 100
  dummy_logits = Parameter(torch.randn(2,3))
  plackett_luce = PlackettLuce(dummy_logits)

  optimizer = torch.optim.Adam([dummy_logits],lr=0.01)

  for _ in range(1000):
    optimizer.zero_grad()
    sampled_indices = plackett_luce.sample((montecarlo_samples,))
    log_probs = plackett_luce.log_prob(sampled_indices)

    pred_relevances = vect_gather(relevances,1,sampled_indices)

    utility = -batched_v_ndcg(pred_relevances)*log_probs

    utility.mean().backward()
    optimizer.step()
  
  #the dummy logits should be increasing for the increasing relevances and decreasing for the others
  dummy_increasing, dummy_decreasing = dummy_logits[0], dummy_logits[1]

  assert all([(dummy_increasing[i] <= dummy_increasing[i+1]).item() for i in range(2)])
  assert all([(dummy_decreasing[i] >= dummy_decreasing[i+1]).item() for i in range(2)])
