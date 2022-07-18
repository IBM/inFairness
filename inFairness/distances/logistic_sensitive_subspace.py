from typing import Iterable
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from inFairness.distances import SensitiveSubspaceDistance
from inFairness.utils import datautils, validationutils


class LogisticRegSensitiveSubspace(SensitiveSubspaceDistance):
    """Implements the Softmax Regression model based fair metric as defined in Appendix B.1
    of "Training individually fair ML models with sensitive subspace robustness" paper.

    This metric assumes that the sensitive attributes are discrete and observed for a small subset
    of training data. Assuming data of the form :math:`(X_i, K_i, Y_i)` where :math:`K_i` is the
    sensitive attribute of the i-th subject, the model fits a softmax regression model to the data as:

    .. math:: \mathbb{P}(K_i = l\\mid X_i) = \\frac{\exp(a_l^TX_i+b_l)}{\\sum_{l=1}^k \\exp(a_l^TX_i+b_l)},\\ l=1,\\ldots,k

    Using the span of the matrix :math:`A=[a_1, \cdots, a_k]`, the fair metric is trained as:

    .. math:: d_x(x_1,x_2)^2 = (x_1 - x_2)^T(I - P_{\\text{ran}(A)})(x_1 - x_2)

    References
    -------------
        `Yurochkin, Mikhail, Amanda Bower, and Yuekai Sun. "Training individually fair
        ML models with sensitive subspace robustness." arXiv preprint arXiv:1907.00020 (2019).`
    """

    def __init__(self):
        super().__init__()
        self.basis_vectors_ = None

    def fit(
        self,
        data_X: torch.Tensor,
        data_SensitiveAttrs: torch.Tensor = None,
        protected_idxs: Iterable[int] = None,
        keep_protected_idxs: bool = True,
        autoinfer_device: bool = True,
    ):
        """Fit Logistic Regression Sensitive Subspace distance metric

        Parameters
        --------------
            data_X: torch.Tensor
                Input data corresponding to either :math:`X_i` or :math:`(X_i, K_i)` in the equation above.
                If the variable corresponds to :math:`X_i`, then the `y_train` parameter should be specified.
                If the variable corresponds to :math:`(X_i, K_i)` then the `protected_idxs` parameter
                should be specified to indicate the sensitive attributes.

            data_SensitiveAttrs: torch.Tensor
                Represents the sensitive attributes ( :math:`K_i` ) and is used when the `X_train` parameter
                represents :math:`X_i` from the equation above. **Note**: This parameter is mutually exclusive
                with the `protected_idxs` parameter. Specififying both the `data_SensitiveAttrs` and `protected_idxs`
                parameters will raise an error

            protected_idxs: Iterable[int]
                If the `X_train` parameter above represents :math:`(X_i, K_i)`, then this parameter is used
                to provide the indices of sensitive attributes in `X_train`. **Note**: This parameter is mutually exclusive
                with the `protected_idxs` parameter. Specififying both the `data_SensitiveAttrs` and `protected_idxs`
                parameters will raise an error

            keep_protected_indices: bool
                True, if while training the model, protected attributes will be part of the training data
                Set to False, if for training the model, protected attributes will be excluded
                Default = True

            autoinfer_device: bool
                Should the distance metric be automatically moved to an appropriate
                device (CPU / GPU) or not? If set to True, it moves the metric
                to the same device `X_train` is on. If set to False, keeps the metric
                on CPU.
        """

        if data_SensitiveAttrs is not None and protected_idxs is None:
            basis_vectors_ = self.compute_basis_vectors_data(
                X_train=data_X, y_train=data_SensitiveAttrs
            )

        elif data_SensitiveAttrs is None and protected_idxs is not None:
            basis_vectors_ = self.compute_basis_vectors_protected_idxs(
                data_X,
                protected_idxs=protected_idxs,
                keep_protected_idxs=keep_protected_idxs,
            )

        else:
            raise AssertionError(
                "Parameters `y_train` and `protected_idxs` are exclusive. Either of these two parameters should be None, and cannot be set to non-None values simultaneously."
            )

        super().fit(basis_vectors_)
        self.basis_vectors_ = basis_vectors_

        if autoinfer_device:
            device = datautils.get_device(data_X)
            super().to(device)

    def compute_basis_vectors_protected_idxs(
        self, data, protected_idxs, keep_protected_idxs=True
    ):

        dtype = data.dtype

        data = datautils.convert_tensor_to_numpy(data)
        basis_vectors_ = []
        num_attr = data.shape[1]

        # Get input data excluding the protected attributes
        protected_idxs = sorted(protected_idxs)
        free_idxs = [idx for idx in range(num_attr) if idx not in protected_idxs]
        X_train = data[:, free_idxs]
        Y_train = data[:, protected_idxs]

        self.__assert_sensitiveattrs_binary__(Y_train)

        coefs = np.array(
            [
                LogisticRegression(solver="liblinear", penalty="l1")
                .fit(X_train, Y_train[:, idx])
                .coef_.squeeze()
                for idx in range(len(protected_idxs))
            ]
        )  # ( |protected_idxs|, |free_idxs| )

        if keep_protected_idxs:
            # To keep protected indices, we add two basis vectors
            # First, with logistic regression coefficients with 0 in
            # protected indices. Second, with one-hot vectors with 1 in
            # protected indices.

            basis_vectors_ = np.empty(shape=(2 * len(protected_idxs), num_attr))

            for i, protected_idx in enumerate(protected_idxs):

                protected_basis_vector = np.zeros(shape=(num_attr))
                protected_basis_vector[protected_idx] = 1.0

                unprotected_basis_vector = np.zeros(shape=(num_attr))
                np.put_along_axis(
                    unprotected_basis_vector, np.array(free_idxs), coefs[i], axis=0
                )

                basis_vectors_[2 * i] = unprotected_basis_vector
                basis_vectors_[2 * i + 1] = protected_basis_vector
        else:
            # Protected indices are to be discarded. Therefore, we can
            # simply return back the logistic regression coefficients
            basis_vectors_ = coefs

        basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
        basis_vectors_ = basis_vectors_.detach()

        return basis_vectors_

    def compute_basis_vectors_data(self, X_train, y_train):

        dtype = X_train.dtype

        X_train = datautils.convert_tensor_to_numpy(X_train)
        y_train = datautils.convert_tensor_to_numpy(y_train)

        self.__assert_sensitiveattrs_binary__(y_train)

        basis_vectors_ = []
        outdim = y_train.shape[-1]

        basis_vectors_ = np.array(
            [
                LogisticRegression(solver="liblinear", penalty="l1")
                .fit(X_train, y_train[:, idx])
                .coef_.squeeze()
                for idx in range(outdim)
            ]
        )

        basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
        basis_vectors_ = basis_vectors_.detach()

        return basis_vectors_

    def __assert_sensitiveattrs_binary__(self, sensitive_attrs):

        assert validationutils.is_tensor_binary(
            sensitive_attrs
        ), "Sensitive attributes are required to be binary to learn the metric. Please binarize these attributes before fitting the metric."
