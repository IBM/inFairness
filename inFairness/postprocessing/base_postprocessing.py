import torch

from typing import Tuple
from inFairness.postprocessing.data_ds import PostProcessingDataStore


class BasePostProcessing(object):
    """Base class for Post-Processing methods

    Parameters
    -------------
        distance_x: inFairness.distances.Distance
            Distance matrix in the input space
        is_output_probas: bool
            True if the `data_Y` (model output) are probabilities implying that
            this is a classification setting, and False if the `data_Y` are
            in euclidean space implying that this is a regression setting.
    """

    def __init__(self, distance_x, is_output_probas):

        self.distance_x = distance_x
        self.is_output_probas = is_output_probas
        self.datastore = PostProcessingDataStore(distance_x)

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

    def __get_yhat__(self):

        _, data_y = self.data

        if self.is_output_probas:
            y_hat = torch.log(data_y[:, :-1]) - torch.log(data_y[:, -1]).view(-1, 1)
            return y_hat
        else:
            return data_y
