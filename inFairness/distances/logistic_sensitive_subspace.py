from typing import Iterable
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from inFairness.distances import SensitiveSubspaceDistance
from inFairness.utils import datautils


class LogisticRegSensitiveSubspace(SensitiveSubspaceDistance):
    """
    does a logistic regression on protected attributes and adds the resulting coefficients as
    a sensitive direction in A. It also adds unitary vectors to A for each protected attribute.
    """

    def __init__(self):
        super().__init__()
        self.basis_vectors_ = None

    def fit(
        self,
        X_train: torch.Tensor,
        protected_idxs: Iterable[int],
        keep_protected_idxs: bool = True,
        autoinfer_device: bool = True,
    ):
        """Fit Logistic Regression Sensitive Subspace distance metric

        Parameters
        --------------
            dataset: torch.Tensor
                the dataset including column names. The dataset should already be
                pre-processed for a logistic regression, columns must be in alphabetical order to
                keep consistency between the feature space computed by the distance and the main
                model.

            protected_idxs: Iterable[int]
                list of indices of the attributes that should be protected,
                e.g `sex` in a dataset to predict `income`. For each of this attributes two basis
                vectors are added: one doing a logistic regression on the remaining attributes, and
                a unit vector for the attribute itself. Only categorical protected attributes supported.

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

        basis_vectors_ = self.compute_basis_vectors(
            X_train, protected_idxs, keep_protected_idxs
        )
        super().fit(basis_vectors_)
        self.basis_vectors_ = basis_vectors_

        if autoinfer_device:
            device = datautils.get_device(X_train)
            super().to(device)

    def compute_basis_vectors(self, data, protected_idxs, keep_protected_idxs=True):

        dtype = data.dtype

        data = datautils.convert_tensor_to_numpy(data)
        basis_vectors_ = []
        num_attr = data.shape[1]

        # Get input data excluding the protected attributes
        protected_idxs = sorted(protected_idxs)
        free_idxs = [idx for idx in range(num_attr) if idx not in protected_idxs]
        X_train = data[:, free_idxs]
        Y_train = data[:, protected_idxs]

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
