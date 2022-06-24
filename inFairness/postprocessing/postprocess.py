import torch

from inFairness.utils.postprocessing import (
    build_graph_from_dists,
    get_laplacian,
    laplacian_solve,
)
from inFairness.postprocessing.base_postprocessing import BasePostProcessing


class GraphLaplacianIF(BasePostProcessing):
    """Implements the Graph Laplacian Individual Fairness Post-Processing method.

    Proposed in `Post-processing for Individual Fairness <https://arxiv.org/abs/2110.13796>`_

    Parameters
    ------------
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
    """

    def __init__(self, distance_x):
        super().__init__(distance_x)

    def __postprocess__(self, y_hat, lambda_param, scale, threshold, normalize):

        W, idxs = build_graph_from_dists(
            self.distance_matrix, scale, threshold, normalize
        )

        data_y_idxs = y_hat[idxs]
        L = get_laplacian(W, normalize)

        if normalize:
            L = (L + L.T) / 2

        y = laplacian_solve(L, data_y_idxs, lambda_param)
        data_y_new = torch.clone(y_hat)
        data_y_new[idxs] = y

        return data_y_new

    def postprocess(
        self, lambda_param: float, scale: float, threshold: float, normalize: bool
    ):
        """Implements the Graph Laplacian Individual Fairness Post-processing algorithm

        Parameters
        -------------
            lambda_param: float
                Weight for the Laplacian Regularizer
            scale: float
                Parameter used to scale the computed distances.
                Refer equation 2.2 in the proposing paper.
            threshold: float
                Parameter used to construct the Graph from distances
                Distances below provided threshold are considered to be
                connected edges, while beyond the threshold are considered to
                be disconnected. Refer equation 2.2 in the proposing paper.
            normalize: bool
                Whether to normalize the computed Laplacian or not

        Returns
        -----------
            solution: torch.Tensor
                post-processed output probabilities of shape (N, C)
                where N is the number of data samples, and C is the
                number of output classes
        """

        _, data_y = self.data
        y_hat = torch.log(data_y[:, :-1]) - torch.log(data_y[:, -1]).view(-1, 1)
        data_y_new = self.__postprocess__(
            y_hat, lambda_param, scale, threshold, normalize
        )
        pp_sol = torch.exp(data_y_new) / (
            1 + torch.exp(data_y_new).sum(axis=1, keepdim=True)
        )
        pp_sol = torch.hstack((pp_sol, 1 - pp_sol.sum(axis=1, keepdim=True)))
        return pp_sol
