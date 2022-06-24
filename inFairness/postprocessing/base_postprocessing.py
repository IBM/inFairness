import torch

from typing import Tuple
from inFairness.postprocessing.data_ds import Data


class BasePostProcessing(object):
    """Base class for Post-Processing methods

    Parameters
    -------------
        distance_x: inFairness.distances.Distance
            Distance matrix in the input space
    """

    def __init__(self, distance_x):

        self.distance_x = distance_x
        self.datastore = Data(distance_x)

    @property
    def data(self):
        """Input and Output data used for post-processing

        Returns
        --------
            data: Tuple(torch.Tensor, torch.Tensor)
                A tuple of (X, Y) data points
        """
        return (self.datastore.data_X, self.datastore.data_Y)

    @property
    def distance_matrix(self):
        """Distance matrix

        Returns
        --------
            distance_matrix: torch.Tensor
                Matrix of distances of shape (N, N) where
                N is the number of data samples
        """
        return self.datastore.distance_matrix

    def add_datapoints(self, X: torch.Tensor, y: torch.Tensor):
        """Add datapoints to the post-processing method

        Parameters
        -----------
            X: torch.Tensor
                New input datapoints
            y: torch.Tensor
                New output datapoints
        """
        self.datastore.add_datapoints(X, y)

    def reset_datapoints(self):
        """Reset datapoints store back to its initial state"""
        self.datastore.reset()

    def postprocess(self, *args, **kwargs):
        raise NotImplementedError("postprocess method not implemented by class")
