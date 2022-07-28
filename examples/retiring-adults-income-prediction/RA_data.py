import torch

from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


def __remove_protected_feats__(data_X, protected_vars):

    features = ACSIncome.features
    protected_idxs = [i for i in range(len(features)) if features[i] in protected_vars]
    unprotected_idxs = [i for i in range(len(features)) if i not in protected_idxs]

    data_X_protected = data_X[:, protected_idxs]
    data_X_unprotected = data_X[:, unprotected_idxs]

    return data_X_unprotected, data_X_protected


def __fetch_data__(state):

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    statedata = data_source.get_data(states=[state], download=True)
    data_X, data_Y, data_grp = ACSIncome.df_to_numpy(statedata)
    return data_X, data_Y, data_grp


def convert_to_tensor(data_X, data_Y):
    data_X = torch.tensor(data_X).float()
    data_Y = torch.tensor(data_Y).long()    
    return data_X, data_Y


def load_data(train_statename, protected_vars):

    train_data = None
    test_data = {}
    scaler = StandardScaler()
    onehot = OneHotEncoder(drop='first')

    # Get train state data
    data_X, data_Y, data_grp = __fetch_data__(train_statename)
    data_X_train, data_X_test, data_Y_train, data_Y_test, _, data_grp_test = train_test_split(
        data_X, data_Y, data_grp, test_size=0.2, random_state=0
    )

    data_X_train, data_X_train_protected = __remove_protected_feats__(data_X_train, protected_vars)
    data_X_test, _ = __remove_protected_feats__(data_X_test, protected_vars)

    # convert protected data to one-hot encoding
    data_X_train_protected = onehot.fit_transform(data_X_train_protected).toarray()
    data_X_train_protected = torch.tensor(data_X_train_protected).long()
    
    # fit standard scaler on training data
    data_X_train = scaler.fit_transform(data_X_train)
    data_X_test = scaler.transform(data_X_test)

    data_X_train, data_Y_train  = convert_to_tensor(data_X_train, data_Y_train)
    data_X_test, data_Y_test = convert_to_tensor(data_X_test, data_Y_test)

    train_data = (data_X_train, data_X_train_protected, data_Y_train)
    test_data[train_statename] = (data_X_test, data_Y_test, data_grp_test)


    # Get other state data

    for state in state_list:
        if state == train_statename:
            continue

        data_X, data_Y, data_grp = __fetch_data__(state)
        data_X, _ = __remove_protected_feats__(data_X, protected_vars)
        data_X = scaler.transform(data_X)
        data_X, data_Y = convert_to_tensor(data_X, data_Y)
        test_data[state] = (data_X, data_Y, data_grp)

    return train_data, test_data
