from typing import Dict
import torch
from dataclasses import dataclass


@dataclass
class PostProcessingObjectiveResponse:
    """Class to store the result from a post-processing algorithm"""

    y_solution: torch.Tensor = None
    objective: Dict = None
