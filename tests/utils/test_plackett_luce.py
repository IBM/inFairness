
import torch
from torch.nn.parameter import Parameter
from inFairness.utils.plackett_luce import PlackettLuce

def test_plackett_luce():
  dummy_logits = Parameter(torch.randn(10))