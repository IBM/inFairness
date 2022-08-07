import torch
import numpy as np

from inFairness.utils.postprocessing import (
    build_graph_from_dists,
    get_laplacian,
    laplacian_solve,
)
from inFairness.postprocessing.base_postprocessing import BasePostProcessing
from inFairness.postprocessing.datainterfaces import PostProcessingObjectiveResponse


class GraphLaplacianIF(BasePostProcessing):
    """Implements the Graph Laplacian Individual Fairness Post-Processing method.

    Proposed in `Post-processing for Individual Fairness <https://arxiv.org/abs/2110.13796>`_

    Parameters
    ------------
        distance_x: inFairness.distances.Distance
            Distance metric in the input space
        is_output_probas: bool
            True if the `data_Y` (model output) are probabilities implying that
            this is a classification setting, and False if the `data_Y` are
            in euclidean space implying that this is a regression setting.
    """

    def __init__(self, distance_x, is_output_probas):

        super().__init__(distance_x, is_output_probas=is_output_probas)
        self._METHOD_COORDINATE_KEY = "coordinate-descent"
        self._METHOD_EXACT_KEY = "exact"

    def __exact_pp__(self, lambda_param, scale, threshold, normalize):
        """Implements Exact version of post processing"""

        y_hat = self.__get_yhat__()

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

        objective = self.get_objective(
            data_y_new, lambda_param, scale, threshold, normalize, W, idxs, L
        )

        return data_y_new, objective

    def __coordinate_update__(
        self,
        yhat_batch,
        W_batch,
        y,
        batchidx,
        lambda_param,
        D_inv_batch=None,
        diag_W_batch=None,
        D_batch=None,
    ):

        W_xy = W_batch.unsqueeze(-1) * y.unsqueeze(0)

        """
        Shapes:
            W_batch: (bsz, nsamples)
            y: (nsamples, ncls -  1)
            W_xy: (bsz, nsamples, ncls-1)
            W_xy_corr: (bsz, ncls-1)
            numerator: (bsz, ncls-1)
            denominator: (bsz, 1)
        """

        if D_inv_batch is None:
            W_xy_corr = torch.diagonal(W_xy[:, batchidx], offset=0, dim1=0, dim2=1).T
            numerator = yhat_batch + lambda_param * (W_xy.sum(dim=1) - W_xy_corr)
            denominator = 1 + lambda_param * (
                W_batch.sum(dim=1, keepdim=True) - diag_W_batch.view(-1, 1)
            )
            y_new = numerator / denominator

        else:
            W_xy = W_xy * D_inv_batch.unsqueeze(-1)
            W_xy_corr = torch.diagonal(W_xy[:, batchidx], offset=0, dim1=0, dim2=1).T
            numerator = (yhat_batch + lambda_param * (W_xy.sum(dim=1) - W_xy_corr)) / 2
            denominator = (
                1
                + lambda_param
                - lambda_param * diag_W_batch.view(-1, 1) / D_batch.view(-1, 1)
            )
            y_new = numerator / denominator

        return y_new

    def __coordinate_pp__(
        self, lambda_param, scale, threshold, normalize, batchsize, epochs
    ):
        """Implements coordinate descent for large-scale data"""

        y_hat = self.__get_yhat__()
        y_copy = y_hat.clone()
        n_samples = self.datastore.n_samples

        W, idxs = build_graph_from_dists(
            self.distance_matrix, scale, threshold, normalize
        )

        data_y_idxs = y_hat[idxs]
        W_diag = torch.diag(W)

        if normalize:
            D = W.sum(dim=1)
            D_inv = 1 / D.reshape(1, -1) + 1 / D.reshape(-1, 1)

        for epoch_idx in range(epochs):
            idxs = np.random.permutation(n_samples)
            curridx = 0

            while curridx < n_samples:
                batchidxs = idxs[curridx : curridx + batchsize]

                if normalize:
                    y_copy[batchidxs] = self.__coordinate_update__(
                        data_y_idxs[batchidxs],
                        W[batchidxs],
                        y_copy,
                        batchidxs,
                        lambda_param=lambda_param,
                        D_inv_batch=D_inv[batchidxs],
                        diag_W_batch=W_diag[batchidxs],
                        D_batch=D[batchidxs],
                    )
                else:
                    y_copy[batchidxs] = self.__coordinate_update__(
                        data_y_idxs[batchidxs],
                        W[batchidxs],
                        y_copy,
                        batchidxs,
                        lambda_param=lambda_param,
                        diag_W_batch=W_diag[batchidxs],
                    )

                curridx += batchsize

        pp_sol = y_hat.clone()
        pp_sol[idxs] = y_copy

        objective = self.get_objective(
            pp_sol, lambda_param, scale, threshold, normalize, W, idxs
        )

        return pp_sol, objective

    def get_objective(
        self,
        y_solution,
        lambda_param: float,
        scale: float,
        threshold: float,
        normalize: bool = False,
        W_graph=None,
        idxs=None,
        L=None,
    ):
        """Compute the objective values for the individual fairness as follows:

        .. math:: \\widehat{\\mathbf{f}} =  \\arg \\min_{\\mathbf{f}} \\ \\|\\mathbf{f} - \\hat{\\mathbf{y}}\\|_2^2 + \\lambda \\ \\mathbf{f}^{\\top}\\mathbb{L_n} \\mathbf{f}

        Refer equation 3.1 in the paper

        Parameters
        ------------
            y_solution: torch.Tensor
                Post-processed solution values of shape (N, C)
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
            W_graph: torch.Tensor
                Adjacency matrix of shape (N, N)
            idxs: torch.Tensor
                Indices of data points which are included in the adjacency matrix
            L: torch.Tensor
                Laplacian of the adjacency matrix

        Returns
        ---------
            objective: PostProcessingObjectiveResponse
                post-processed solution containing two parts:
                    (a) Post-processed output probabilities of shape  (N, C)
                        where N is the number of data samples, and C is the
                        number of output classes
                    (b) Objective values. Refer equation 3.1 in the paper
                        for an explanation of the various parts

        """

        if W_graph is None or idxs is None:
            W_graph, idxs = build_graph_from_dists(
                self.distance_matrix, scale, threshold, normalize
            )
        if L is None:
            L = get_laplacian(W_graph, normalize)

        y_hat = self.__get_yhat__()
        y_dist = ((y_hat - y_solution) ** 2).sum()
        L_obj = lambda_param * (y_solution[idxs] * (L @ y_solution[idxs])).sum()
        overall_objective = y_dist + L_obj

        result = {
            "y_dist": y_dist.item(),
            "L_objective": L_obj.item(),
            "overall_objective": overall_objective.item(),
        }

        return result

    def postprocess(
        self,
        method: str,
        lambda_param: float,
        scale: float,  # 0.001
        threshold: float,  # median of all distances if None
        normalize: bool = False,
        batchsize: int = None,
        epochs: int = None,
    ):
        """Implements the Graph Laplacian Individual Fairness Post-processing algorithm

        Parameters
        -------------
            method: str
                GLIF method type. Possible values are:

                (a) `coordinate-descent` method which is more suitable for
                large-scale data and post-processes by batching data into minibatches
                (see section 3.2.2 of the paper), or

                (b) `exact` method which gets the exact solution but is not appropriate
                for large-scale data (refer equation 3.3 in the paper).
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
            batchsize: int
                Batch size. *Required when method=`coordinate-descent`*
            epochs: int
                Number of coordinate descent epochs.
                *Required when method=`coordinate-descent`*

        Returns
        -----------
            solution: PostProcessingObjectiveResponse
                post-processed solution containing two parts:
                    (a) Post-processed output probabilities of shape  (N, C)
                        where N is the number of data samples, and C is the
                        number of output classes
                    (b) Objective values. Refer equation 3.1 in the paper
                        for an explanation of the various parts
        """

        assert method in [
            self._METHOD_COORDINATE_KEY,
            self._METHOD_EXACT_KEY,
        ], f"`method` should be either `coordinate-descent` or `exact`. Value provided: {method}"

        if method == self._METHOD_COORDINATE_KEY:
            assert (
                batchsize is not None and epochs is not None
            ), f"batchsize and epochs parameter is required but None provided"

        if method == self._METHOD_EXACT_KEY:
            data_y_new, objective = self.__exact_pp__(
                lambda_param, scale, threshold, normalize
            )
        elif method == self._METHOD_COORDINATE_KEY:
            data_y_new, objective = self.__coordinate_pp__(
                lambda_param, scale, threshold, normalize, batchsize, epochs
            )

        if self.is_output_probas:
            pp_sol = torch.exp(data_y_new) / (
                1 + torch.exp(data_y_new).sum(axis=1, keepdim=True)
            )
            y_solution = torch.hstack((pp_sol, 1 - pp_sol.sum(axis=1, keepdim=True)))
        else:
            y_solution = data_y_new

        result = PostProcessingObjectiveResponse(
            y_solution=y_solution, objective=objective
        )
        return result
