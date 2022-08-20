import torch


class DistanceStructure(object):
    """Data structure to store and track the distance matrix between data points

    Parameters
    -------------
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
    """

    def __init__(self, distance_x):
        self.distance_x = distance_x
        self.distance_matrix = None

    def reset(self):
        """Reset the state of the data structure back to its initial state"""
        self.distance_matrix = None

    def build_distance_matrix(self, data_X):
        """Build the distance matrix between input data samples `data_X`

        Parameters
        -------------
            data_X: torch.Tensor
                Data points between which the distance matrix is to be computed
        """

        nsamples_old = (
            0 if self.distance_matrix is None else self.distance_matrix.shape[0]
        )
        nsamples_total = data_X.shape[0]
        device = data_X.device

        distance_matrix_new = torch.zeros(
            size=(nsamples_total, nsamples_total), device=device
        )

        if self.distance_matrix is not None:
            distance_matrix_new[:nsamples_old, :nsamples_old] = self.distance_matrix

        dist = (
            self.distance_x(
                data_X[nsamples_old:nsamples_total], data_X, itemwise_dist=False
            )
            .detach()
            .squeeze()
        )
        distance_matrix_new[nsamples_old:, :] = dist
        distance_matrix_new[:, nsamples_old:] = dist.T

        self.distance_matrix = distance_matrix_new.clone()
