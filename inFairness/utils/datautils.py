from typing import Iterable

import torch
import numpy as np

from itertools import product


def generate_data_pairs(n_pairs, datasamples_1, datasamples_2=None, comparator=None):
    """Utility function to generate (in)comparable data pairs given data samples. Use case includes
    creating a dataset of comparable and incomparable data for the EXPLORE distance metric which
    learns from such data samples.

    Parameters
    ------------
        n_pairs: int
                Number of pairs to construct
        datasamples_1: numpy.ndarray
                Array of data samples of shape (N_1, *)
        datasamples_2: numpy.ndarray
                (Optional) array of data samples of shape (N_2, *).
                If datasamples_2 is provided, then data pairs are constructed between
                datasamples_1 and datasamples_2.
                If datasamples_2 is not provided, then data pairs are constructed within
                datasamples_1
        comparator: function
                A lambda function that given two samples returns True if they should
                be paired, and False if not.
                If `comparator` is not defined, then random data samples are paired together.
                Example: `comparator = lambda x, y: (x == y)`

    Returns
    ----------
        idxs: numpy.ndarray
                A (n_pairs, 2) shaped array with indices of data sample pairs
    """

    if datasamples_2 is None:
        datasamples_2 = datasamples_1

    nsamples_1 = datasamples_1.shape[0]
    nsamples_2 = datasamples_2.shape[0]

    if comparator is None:
        ntotal = nsamples_1 * nsamples_2
        assert (
            n_pairs <= ntotal
        ), f"Number of desired data pairs {n_pairs} is greater than possible combinations {ntotal}"

        idxs = np.random.choice(ntotal, n_pairs, replace=False)
        idxs1, idxs2 = np.unravel_index(idxs, shape=(nsamples_1, nsamples_2))
        idxs = np.stack((idxs1, idxs2), axis=-1)
    else:
        all_idxs = [
            (idx1, idx2)
            for idx1, idx2 in product(range(nsamples_1), range(nsamples_2))
            if comparator(datasamples_1[idx1], datasamples_2[idx2])
        ]
        assert n_pairs <= len(all_idxs), (
            f"Number of desired data pairs {n_pairs} is greater than possible "
            + "combinations {len(all_idxs)}"
        )
        idx_positions = np.random.choice(len(all_idxs), n_pairs, replace=False)
        idxs = np.array([all_idxs[x] for x in idx_positions])

    return idxs


def convert_tensor_to_numpy(tensor):
    """Converts a PyTorch tensor to numpy array

    If the provided `tensor` is not a PyTorch tensor, it returns the same object back
    with no modifications

    Parameters
    -----------
        tensor: torch.Tensor
            Tensor to be converted to numpy array

    Returns
    ----------
        array_np: numpy.ndarray
            Numpy array of the provided tensor
    """

    if torch.is_tensor(tensor):
        array_np = tensor.detach().cpu().numpy()
        return array_np

    return tensor


def include_exclude_terms(
    data_terms: Iterable[str], include: Iterable[str] = (), exclude: Iterable[str] = ()
):
    """
    given a set of data terms, return a resulting set depending on specified included and excluded terms.

    Parameters
    -----------
        data_terms: string iterable
                    set of terms to be filtered
        include: string iterable
                 set of terms to be included, if not specified all data_terms are included
        exclude: string iterable
                 set of terms to be excluded from data_terms
    Returns
    ----------
        terms: list of strings
               resulting terms in alphabetical order.
    """
    terms = set(include) if len(include) > 0 else set(data_terms)
    if len(exclude) > 0:
        terms = terms.difference(set(exclude))
    terms = sorted(list(terms))
    return terms


def get_device(obj):
    """Returns a device (cpu/cuda) based on the type of the reference object

    Parameters
    -------------
        obj: torch.Tensor

    """

    device = torch.device("cpu")

    # If reference object is a tensor, use its device
    if torch.is_tensor(obj):
        device = obj.device

    # If reference object is a list, check if first element is a tensor
    # and if it is a tensor, use it's device
    if isinstance(obj, list) and torch.is_tensor(obj[0]):
        device = obj[0].device

    return device
