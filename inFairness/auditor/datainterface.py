import torch

from dataclasses import dataclass


@dataclass
class AuditorResponse:
    """Class to store a result from the auditor"""

    lossratio_mean: float = None
    lossratio_std: float = None
    lower_bound: float = None
    threshold: float = None
    pval: float = None
    confidence: float = None
    is_model_fair: bool = None
