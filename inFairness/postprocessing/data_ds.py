import torch

from inFairness.postprocessing.distance_ds import DistanceStructure


class PostProcessingDataStore(object):
    """Data strucuture to hold the data used for post-processing

    Parameters
    -------------
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
    """

    def __init__(self, distance_x):

        self.data_X = None
        self.data_Y = None
        self.n_samples = 0
        self.distance_ds = DistanceStructure(distance_x)

    @property
    def distance_matrix(self):
        """Distances between N data points. Shape: (N, N)"""
        return self.distance_ds.distance_matrix

    def add_datapoints_X(self, X: torch.Tensor):
        """Add datapoints to the input datapoints X

        Parameters
        ------------
            X: torch.Tensor
                New data points to add to the input data
                `X` should have the same dimensions as previous data
                along all dimensions except the first (batch) dimension
        """

        if self.data_X is None:
            self.data_X = X
        else:
            self.data_X = torch.cat([self.data_X, X], dim=0)

    def add_datapoints_Y(self, Y: torch.Tensor):
        """Add datapoints to the output datapoints Y

        Parameters
        ------------
            Y: torch.Tensor
                New data points to add to the output data
                `Y` should have the same dimensions as previous data
                along all dimensions except the first (batch) dimension
        """

        if self.data_Y is None:
            self.data_Y = Y
        else:
            self.data_Y = torch.cat([self.data_Y, Y], dim=0)

    def add_datapoints(self, X: torch.Tensor, Y: torch.Tensor):
        """Add new datapoints to the existing datapoints

        Parameters
        ------------
            X: torch.Tensor
                New data points to add to the input data
                `X` should have the same dimensions as previous data
                along all dimensions except the first (batch) dimension
            Y: torch.Tensor
                New data points to add to the output data
                `Y` should have the same dimensions as previous data
                along all dimensions except the first (batch) dimension
        """

        self.add_datapoints_X(X)
        self.add_datapoints_Y(Y)
        self.n_samples = self.n_samples + X.shape[0]
        self.distance_ds.build_distance_matrix(self.data_X)

    def reset(self):
        """Reset the data structure holding the data points for post-processing.
        Invoking this operation removes all datapoints and resets the state back
        to the initial state.
        """

        self.data_X = None
        self.data_Y = None
        self.n_samples = 0
        self.distance_ds.reset()
