import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from typing import List

from inFairness.distances.mahalanobis_distance import MahalanobisDistances
from inFairness.utils import datautils


class SensitiveSubspaceDistance(MahalanobisDistances):
    """Implements Sensitive Subspace metric base class that accepts the
    basis vectors of a sensitive subspace, and computes a projection
    that ignores the sensitive subspace.

    The projection from the sensitive subspace basis vectors (A) is computed as:

    .. math:: P^{'} = I - (A (A A^{T})^{-1} A^{T})
    """

    def __init__(self):
        super().__init__()

    def fit(self, basis_vectors):
        """Fit Sensitive Subspace Distance metric

        Parameters
        --------------
            basis_vectors: torch.Tensor
                Basis vectors of the sensitive subspace
        """

        sigma = self.compute_projection_complement(basis_vectors)
        super().fit(sigma)

    def compute_projection_complement(self, basis_vectors):
        """Compute the projection complement of the space
        defined by the basis_vectors:

        projection complement given basis vectors (A) is computed as:

        .. math:: P^{'} = I - (A (A A^{T})^{-1} A^{T})

        Parameters
        -------------
            basis_vectors: torch.Tensor
                Basis vectors of the sensitive subspace
                Dimension (d, k) where d is the data features dimension
                and k is the number of protected dimensions

        Returns
        ----------
            projection complement: torch.Tensor
                Projection complement computed as described above.
                Shape (d, d) where d is the data feature dimension
        """

        # Computing the orthogonal projection
        # V(V V^T)^{-1} V^T
        projection = torch.linalg.inv(torch.matmul(basis_vectors.T, basis_vectors))

        projection = torch.matmul(basis_vectors, projection)

        # Shape: (n_features, n_features)
        projection = torch.matmul(projection, basis_vectors.T)

        # Complement the projection as: (I - Proj)
        projection_complement_ = torch.eye(projection.shape[0]) - projection
        projection_complement_ = projection_complement_.detach()

        return projection_complement_


class SVDSensitiveSubspaceDistance(SensitiveSubspaceDistance):
    """Sensitive Subspace metric that uses SVD to find the basis vectors of
    the sensitive subspace. The metric learns a subspace from a set of
    user-curated comparable data samples.

    Proposed in Section B.2 of Training individually fair ML models
    with sensitive subspace robustness

    References
    -------------
        `Yurochkin, Mikhail, Amanda Bower, and Yuekai Sun. "Training individually fair
        ML models with sensitive subspace robustness." arXiv preprint arXiv:1907.00020 (2019).`
    """

    def __init__(self):
        super().__init__()
        self.n_components_ = None

    def fit(self, X_train, n_components, autoinfer_device=True):
        """Fit SVD Sensitive Subspace distance metric parameters

        Parameters
        -------------
            X_train: torch.Tensor | List[torch.Tensor]
                Training data containing comparable data samples.
                If only one set of comparable data samples is provided, the input
                should be a torch.Tensor of shape :math:`(N, D)`. For multiple sets
                of comparable data samples a list of shape
                :math:`[ (N_1, D), \\cdots, (N_x, D)]` can be provided.
            n_components: int
                Desired number of latent variable dimensions
            autoinfer_device: bool
                Should the distance metric be automatically moved to an appropriate
                device (CPU / GPU) or not? If set to True, it moves the metric
                to the same device `X_train` is on. If set to False, keeps the metric
                on CPU.
        """

        self.n_components_ = n_components
        basis_vectors = self.compute_basis_vectors(X_train, n_components)
        super().fit(basis_vectors)

        if autoinfer_device:
            device = datautils.get_device(X_train)
            super().to(device)

    def __process_input_data__(self, X_train):
        """Process metric training data to convert from tensor to numpy and
        remove the mean and concatenate if multiple sets of training data
        is provided
        """

        if isinstance(X_train, torch.Tensor) or isinstance(X_train, np.ndarray):
            X_train = datautils.convert_tensor_to_numpy(X_train)
            return X_train

        if isinstance(X_train, list):
            X_train = [datautils.convert_tensor_to_numpy(X) for X in X_train]

            # Subtract mean and concatenate all sets of features
            X_norm = np.vstack([X - np.mean(X, axis=0) for X in X_train])
            return X_norm

        raise TypeError(
            "Provided data `X_train` should either be Tensor, np.ndarray or a list of these."
        )

    def compute_basis_vectors(self, X_train, n_components):
        """Compute basis vectors using SVD"""

        X_train = self.__process_input_data__(X_train)
        tSVD = TruncatedSVD(n_components=n_components)
        tSVD.fit(X_train)
        basis_vectors_ = tSVD.components_.T  # Shape: (n_features, n_components)
        basis_vectors_ = torch.Tensor(basis_vectors_)
        return basis_vectors_
