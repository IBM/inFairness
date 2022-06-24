import torch


def build_graph_from_dists(
    dists: torch.Tensor,
    scale: float = None,
    threshold: float = None,
    normalize: bool = False,
):
    """Build the adjacency matrix `W` given distances

    Parameters
    -------------
        dists: torch.Tensor
            Distance matrix between data points. Shape: (N, N)
        scale: float
            Parameter used to scale the computed distances
        threshold: float
            Parameter used to determine if two data points are connected or not.
            Distances below threshold value are connected, and beyond
            threshold value are disconnected.
        normalize: bool
            Whether to normalize the adjacency matrix or not

    Returns
    ----------
        W: torch.Tensor
            Adjancency matrix. It contains data points which are connected
            to atleast one other datapoint. Isolated datapoints, or ones which
            are not connected to any other datapoints, are not included in the
            adjancency matrix.
        idxs_in: torch.Tensor
            Indices of data points which are included in the adjacency matrix
    """

    scale = 1.0 if scale is None else scale
    threshold = 1e10 if threshold is None else threshold

    W = torch.exp(-(dists * scale)) * (torch.sqrt(dists) < threshold)
    idxs_in = torch.where(W.sum(axis=1) > 0.0)[0]

    W = W[idxs_in]
    W = W[:, idxs_in]

    if normalize:
        D_inv_sqrt = 1.0 / torch.sqrt(W.sum(axis=1))
        W = W * D_inv_sqrt * D_inv_sqrt.view(-1, 1)

    return W, idxs_in


def get_laplacian(W: torch.Tensor, normalize: bool = False):
    """Get the Laplacian of the matrix `W`

    Parameters
    -------------
        W: torch.Tensor
            Adjacency matrix of shape (N, N)
        normalize: bool
            Whether to normalize the computed laplacian or not

    Returns
    -------------
        Laplacian: torch.Tensor
            Laplacian of the adjacency matrix
    """

    D = W.sum(axis=1)
    L = torch.diag(D) - W

    if normalize:
        L = L / D.view(-1, 1)

    return L


def laplacian_solve(L: torch.Tensor, y_hat: torch.Tensor, lambda_param: float = None):
    """Solves a system of linear equations to get the post-processed output.
    The system of linear equation it solves is:
    :math:`\hat{{f}} = {(I + \lambda * L)}^{-1} \hat{y}`

    Parameters
    ------------
        L: torch.Tensor
            Laplacian matrix
        y_hat: torch.Tensor
            Model predicted output class probabilities
        lambda_param: float
            Weight for the laplacian regularizer

    Returns
    ----------
        y: torch.Tensor
            Post-processed solution according to the equation above
    """

    lambda_param = 1.0 if lambda_param is None else lambda_param
    n = L.shape[0]
    y = torch.linalg.solve(lambda_param * L + torch.eye(n), y_hat)

    return y
