from abc import ABCMeta, abstractmethod
from torch import nn


class Distance(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for model distances
    """

    def __init__(self):
        super().__init__()

    def fit(self, **kwargs):
        """
        Fits the metric parameters for learnable metrics
        Default functionality is to do nothing. Subclass
        should overwrite this method to implement custom fit
        logic
        """
        pass

    @abstractmethod
    def forward(self, x, y):
        """
        Subclasses must override this method to compute particular distances

        Returns:
             Tensor: distance between two inputs
        """
