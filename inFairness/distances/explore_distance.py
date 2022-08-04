import numpy as np
import torch
from scipy.stats import logistic

from inFairness.utils import datautils
from inFairness.distances.mahalanobis_distance import MahalanobisDistances


class EXPLOREDistance(MahalanobisDistances):
    """Implements the Embedded Xenial Pairs Logistic Regression metric
    (EXPLORE) defined in Section 2.2 of Two Simple Ways to Learn Individual
    Fairness Metrics from Data.

    EXPLORE defines the distance in the input space to be of the form:

    .. math:: d_x(x_1, x_2) := \langle \phi(x_1) - \phi(x_2), \Sigma (\phi(x_1) - \phi(x_2)) \\rangle
    where :math:`\phi(x)` is an embedding map and :math:`\Sigma` is a semi-positive
    definite matrix.

    The metric expects the data to be in the form of triplets
    :math:`\{(x_{i_1}, x_{i_2}, y_i)\}_{i=1}^{n}` where :math:`y_i \in \{0, 1\}`
    indicates whether the human considers :math:`x_{i_1}` and :math:`x_{i_2}`
    comparable (:math:`y_i = 1` indicates comparable) or not.

    References
    -----------
        `Mukherjee, Debarghya, Mikhail Yurochkin, Moulinath Banerjee, and Yuekai Sun.
        "Two simple ways to learn individual fairness metrics from data." In
        International Conference on Machine Learning, pp. 7097-7107. PMLR, 2020.`
    """

    def __init__(self):
        super().__init__()

    def fit(self, X1, X2, Y, iters, batchsize, autoinfer_device=True):
        """Fit EXPLORE distance metric

        Parameters
        -----------
            X1: torch.Tensor
                first set of input samples
            X2: torch.Tensor
                second set of input samples
            Y: torch.Tensor
                :math:`y_i` vector containing 1 if corresponding elements from
                X1 and X2 are comparable, and 0 if not
            iters: int
                number of iterations of SGD to compute the :math:`\Sigma` matrix
            batchsize: int
                batch size of each iteration
            autoinfer_device: bool
                Should the distance metric be automatically moved to an appropriate
                device (CPU / GPU) or not? If set to True, it moves the metric
                to the same device `X1` is on. If set to False, keeps the metric
                on CPU.
        """

        X = datautils.convert_tensor_to_numpy(X1 - X2)
        Y = datautils.convert_tensor_to_numpy(Y)
        sigma = self.compute_sigma(X, Y, iters, batchsize)
        super().fit(sigma)

        if autoinfer_device:
            device = datautils.get_device(X1)
            super().to(device)

    def __grad_likelihood__(self, X, Y, sigma):
        """Computes the gradient of the likelihood function using sigmoidal link"""

        diag = np.einsum("ij,ij->i", np.matmul(X, sigma), X)
        diag = np.maximum(diag, 1e-10)
        prVec = logistic.cdf(diag)
        sclVec = 2.0 / (np.exp(diag) - 1)
        vec = (Y * prVec) - ((1 - Y) * prVec * sclVec)
        grad = np.matmul(X.T * vec, X) / X.shape[0]
        return grad

    def __projPSD__(self, sigma):
        """Computes the projection onto the PSD cone"""

        try:
            L = np.linalg.cholesky(sigma)
            sigma_hat = np.dot(L, L.T)
        except np.linalg.LinAlgError:
            d, V = np.linalg.eigh(sigma)
            sigma_hat = np.dot(
                V[:, d >= 1e-8], d[d >= 1e-8].reshape(-1, 1) * V[:, d >= 1e-8].T
            )
        return sigma_hat

    def compute_sigma(self, X, Y, iters, batchsize):

        N = X.shape[0]
        P = X.shape[1]

        sigma_t = np.random.normal(0, 1, P**2).reshape(P, P)
        sigma_t = np.matmul(sigma_t, sigma_t.T)
        sigma_t = sigma_t / np.linalg.norm(sigma_t)

        curriter = 0

        while curriter < iters:
            batch_idxs = np.random.choice(N, size=batchsize, replace=False)
            X_batch = X[batch_idxs]
            Y_batch = Y[batch_idxs]

            grad_t = self.__grad_likelihood__(X_batch, Y_batch, sigma_t)
            t = 1.0 / (1 + curriter // 100)
            sigma_t = self.__projPSD__(sigma_t - t * grad_t)

            curriter += 1

        sigma = torch.FloatTensor(sigma_t).detach()
        return sigma
