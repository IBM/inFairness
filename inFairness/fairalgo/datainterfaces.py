import torch

from dataclasses import dataclass


@dataclass
class FairModelResponse:
    """Class to store a result from the fairmodel algorithm"""

    loss: torch.Tensor = None
    y_pred: torch.Tensor = None
