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

    def load_state_dict(self, state_dict, strict=True):

        buffer_keys = [bufferitem[0] for bufferitem in self.named_buffers()]
        for key, val in state_dict.items():
            if key not in buffer_keys and strict:
                raise AssertionError(
                    f"{key} not found in metric state and strict parameter is set to True. Either set strict parameter to False or remove extra entries from the state dictionary."
                )
            setattr(self, key, val)

    @abstractmethod
    def forward(self, x, y):
        """
        Subclasses must override this method to compute particular distances

        Returns:
             Tensor: distance between two inputs
        """
