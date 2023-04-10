import os
import requests
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement


def _download_data_(rootdir=None):

    URLS = {
        'train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    }

    dirpaths = {}

    if rootdir is None:
        rootdir = "./dataset"

    os.makedirs(rootdir, exist_ok=True)

    for fname, url in URLS.items():
        fout = os.path.join(rootdir, f'{fname}.csv')

        r = requests.get(url)
        with open(fout, 'w') as f:
            f.write(r.content.decode('utf-8'))

        dirpaths[fname] = fout
    
    return dirpaths


def _read_data_(fpath, train_or_test):

    names = [
        'age', 'workclass', 'fnlwgt', 'education', 
        'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 
        'capital-loss', 'hours-per-week', 'native-country',
        'annual-income'
    ]

    if train_or_test == 'train':
        data = pd.read_csv(
            fpath, sep=',', header=None, names=names,
            na_values=['?'], skipinitialspace=True
        )
    elif train_or_test == 'test':
        data = pd.read_csv(
            fpath, sep=',', header=None, names=names,
            na_values=['?'], skiprows=1, skipinitialspace=True
        )
        data['annual-income'] = data['annual-income'].str.rstrip('.')

    data['annual-income'] = data['annual-income'].replace({'<=50K': 0, '>50K': 1})
    
    return data


def load_data(rootdir=None):

    # download data from UCI repository
    dirpaths = _download_data_(rootdir=rootdir)
    
    train_data = _read_data_(dirpaths['train'], 'train')
    test_data = _read_data_(dirpaths['test'], 'test')

    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # remove rows with NaNs
    data.dropna(inplace=True)

    categorical_vars = [
        'workclass', 'marital-status', 'occupation', 
        'relationship', 'race', 'sex', 'native-country'
    ]

    data = pd.get_dummies(data, columns=categorical_vars)

    cols_to_drop = [
        'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
        'race_Other', 'sex_Female', 'native-country_Cambodia', 'native-country_Canada', 
        'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 
        'native-country_Dominican-Republic', 'native-country_Ecuador', 
        'native-country_El-Salvador', 'native-country_England', 'native-country_France', 
        'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 
        'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 
        'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 
        'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 
        'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 
        'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 
        'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 
        'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 
        'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia',
        'fnlwgt', 'education'
    ]

    data.drop(cols_to_drop, axis=1, inplace=True)

    # Split into train/test splits
    train_data = data.sample(frac=0.8, random_state=123)
    test_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    # Standardize continuous columns
    continuous_vars = [
        'age', 'education-num', 'capital-gain', 
        'capital-loss', 'hours-per-week'
    ]
    scaler = StandardScaler().fit(train_data[continuous_vars])
    train_data[continuous_vars] = scaler.transform(train_data[continuous_vars])
    test_data[continuous_vars] = scaler.transform(test_data[continuous_vars])

    train_data = get_input_output_df(train_data)
    test_data = get_input_output_df(test_data)

    return train_data, test_data


def get_input_output_df(data):

    cols = sorted(data.columns)
    output_col = 'annual-income'
    input_cols = [col for col in cols if col not in output_col]

    df_X = data[input_cols]
    df_Y = data[output_col]
    
    return df_X, df_Y


def convert_df_to_tensor(data_X_df, data_Y_df):

    data_X = torch.tensor(data_X_df.values).float()
    data_Y = torch.tensor(data_Y_df.values)

    return data_X, data_Y


def generate_pairs(len1, len2, n_pairs=100):
    """
    vanilla sampler of random pairs (might sample same pair up to permutation)
    n_pairs > len1*len2 should be satisfied
    """
    idx = sample_without_replacement(len1*len2, n_pairs)
    return np.vstack(np.unravel_index(idx, (len1, len2)))


def create_data_pairs(X_train, Y_train, Y_gender_train, n_comparable=10000, n_incomparable=10000):
    y_gender_train_np = Y_gender_train.detach().numpy()
    y_train_np = Y_train.detach().numpy()
    X_train_np = X_train.detach().numpy()

    # Create comparable pairs
    comparable_X1 = None
    comparable_X2 = None

    K = 2
    for i in range(K):
        c0_idx = np.where((1*(y_gender_train_np==0) + (y_train_np==i))==2)[0]
        c1_idx = np.where((1*(y_gender_train_np==1) + (y_train_np==i))==2)[0]
        pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=n_comparable // K)
        if comparable_X1 is None:
            comparable_X1 = X_train_np[c0_idx[pairs_idx[0]]]
            comparable_X2 = X_train_np[c1_idx[pairs_idx[1]]]
        else:
            comparable_X1 = np.vstack((comparable_X1, X_train_np[c0_idx[pairs_idx[0]]]))
            comparable_X2 = np.vstack((comparable_X2, X_train_np[c1_idx[pairs_idx[1]]]))

    # Create incomparable pairs
    c0_idx = np.where(y_train_np==0)[0]
    c1_idx = np.where(y_train_np==1)[0]
    pairs_idx = generate_pairs(len(c0_idx), len(c1_idx), n_pairs=n_incomparable)

    incomparable_X1 = X_train_np[c0_idx[pairs_idx[0]]]
    incomparable_X2 = X_train_np[c1_idx[pairs_idx[1]]]


    # Join the two sets (comparable and incomparable) to create X and Y
    X1 = np.vstack((comparable_X1, incomparable_X1))
    X2 = np.vstack((comparable_X2, incomparable_X2))

    Y_pairs = np.zeros(n_comparable + n_incomparable)
    Y_pairs[:n_comparable] = 1

    X1 = torch.from_numpy(X1)
    X2 = torch.from_numpy(X2)
    Y_pairs = torch.from_numpy(Y_pairs)

    return X1, X2, Y_pairs
