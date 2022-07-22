import torch
import inFairness.utils.normalized_discounted_cumulative_gain as ndcg

def test_normalized_discounted_cumulative_gain():
  x = torch.tensor([10,8.,1.])
  assert ndcg.normalized_discounted_cumulative_gain(x) == 1.0

  x = torch.tensor([1.,2,3])

  assert ndcg.normalized_discounted_cumulative_gain(x) - 0.7397 < 0.01

  batch_x = torch.arange(8,dtype=torch.float).reshape(2,4)
  assert (ndcg.vect_normalized_discounted_cumulative_gain(batch_x) - 0.6447 < 1e-2).all()

  batch_x,_ = torch.sort(batch_x, descending=True, dim=1)
  assert (ndcg.vect_normalized_discounted_cumulative_gain(batch_x) - 1. < 1e-2).all()
