import torch
import numpy as np

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

        self._METHOD_COORDINATE_KEY = "coordinate-descent"
        self._METHOD_EXACT_KEY = "exact"

    def __get_yhat__(self):
        _, data_y = self.data
        y_hat = torch.log(data_y[:, :-1]) - torch.log(data_y[:, -1]).view(-1, 1)
        return y_hat

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

        return data_y_new

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

        tmp = {
            "yhat_batch": yhat_batch,
            "w_batch": W_batch,
            "y": y,
            "diag_W_batch": diag_W_batch,
            "D_inv_batch": D_inv_batch,
            "D_batch": D_batch,
        }

        for k, v in tmp.items():
            if v is not None:
                print(k, "-->", v.shape)

        """
        Shapes:
            W_batch: (bsz, nsamples)
            y: (nsamples, ncls -  1)
            W_xy: (bsz, nsamples, ncls-1)
            W_xy_corr: (bsz, ncls-1)
            numerator: (bsz, ncls-1)
            denominator: (bsz, 1)
        """
        W_xy = W_batch.unsqueeze(-1) * y.unsqueeze(0)

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
        print("y_copy: ", y_copy.shape)
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
        return pp_sol

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
            solution: torch.Tensor
                post-processed output probabilities of shape (N, C)
                where N is the number of data samples, and C is the
                number of output classes
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
            data_y_new = self.__exact_pp__(lambda_param, scale, threshold, normalize)
        elif method == self._METHOD_COORDINATE_KEY:
            data_y_new = self.__coordinate_pp__(
                lambda_param, scale, threshold, normalize, batchsize, epochs
            )

        pp_sol = torch.exp(data_y_new) / (
            1 + torch.exp(data_y_new).sum(axis=1, keepdim=True)
        )
        pp_sol = torch.hstack((pp_sol, 1 - pp_sol.sum(axis=1, keepdim=True)))
        return pp_sol
