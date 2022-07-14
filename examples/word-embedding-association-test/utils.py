import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils.random import sample_without_replacement
from itertools import combinations

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def statistics(X, Y, A, B):

    score_x = (
        np.matmul(X, A.T).mean(axis=1).sum() - np.matmul(X, B.T).mean(axis=1).sum()
    )
    score_y = (
        np.matmul(Y, A.T).mean(axis=1).sum() - np.matmul(Y, B.T).mean(axis=1).sum()
    )

    return np.abs(score_x - score_y)


def effect_size(X, Y, A, B, Sigma=None):

    mean_x = (np.matmul(X, A.T).mean(axis=1) - np.matmul(X, B.T).mean(axis=1)).mean()
    mean_y = (np.matmul(Y, A.T).mean(axis=1) - np.matmul(Y, B.T).mean(axis=1)).mean()

    XY = np.vstack((X, Y))
    std_xy = (np.matmul(XY, A.T).mean(axis=1) - np.matmul(XY, B.T).mean(axis=1)).std()

    return np.abs(mean_x - mean_y) / std_xy


def normalize_list(embeds, Sigma=None, proj=None):
    res = []
    for e in embeds:
        if proj is not None:
            e = np.matmul(e, proj)
        if Sigma is None:
            res.append(normalize(e))
        else:
            sigma_norm = np.sqrt(np.einsum("ij,ij->i", np.matmul(e, Sigma), e))
            res.append(e / sigma_norm.reshape(-1, 1))
    return res


def run_test(X, Y, A, B, Sigma=None, proj=None, n_combinations=50000):
    X, Y, A, B = normalize_list([X, Y, A, B], Sigma=Sigma, proj=proj)
    if Sigma is not None:
        A = np.matmul(A, Sigma)
        B = np.matmul(B, Sigma)

    base_statistics = statistics(X, Y, A, B)

    union_XY = np.vstack((X, Y))
    xy_size = union_XY.shape[0]
    x_size = X.shape[0]
    count = 0

    all_idx = set(range(xy_size))

    if comb(xy_size, x_size) > n_combinations:
        for _ in range(n_combinations):
            group_1_idx = sample_without_replacement(xy_size, x_size)
            group_2_idx = list(all_idx.difference(group_1_idx))
            sample_stat = statistics(union_XY[group_1_idx], union_XY[group_2_idx], A, B)
            count += sample_stat > base_statistics
    else:
        for group_1_idx in combinations(range(xy_size), x_size):
            group_2_idx = list(all_idx.difference(group_1_idx))
            sample_stat = statistics(
                union_XY[list(group_1_idx)], union_XY[group_2_idx], A, B
            )
            count += sample_stat > base_statistics

    p_val = count / n_combinations
    effect_val = effect_size(X, Y, A, B)
    return p_val, effect_val
