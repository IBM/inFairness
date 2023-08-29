import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.random import sample_without_replacement
from sklearn.decomposition import TruncatedSVD

def load_embeddings(filename):
    """
    Load a DataFrame from the generalized text format used by word2vec, GloVe,
    fastText, and ConceptNet Numberbatch. The main point where they differ is
    whether there is an initial line with the dimensions of the matrix.
    """
    labels = []
    rows = []
    with open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            items = line.rstrip().split(' ')
            if len(items) == 2:
                # This is a header row giving the shape of the matrix
                continue
            labels.append(items[0])
            values = np.array([float(x) for x in items[1:]], 'f')
            rows.append(values)
    
    arr = np.vstack(rows)
    return pd.DataFrame(arr, index=labels, dtype='f')


def load_lexicon(filename):
    """
    Load a file from Bing Liu's sentiment lexicon
    (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
    English words in Latin-1 encoding.
    
    One file contains a list of positive words, and the other contains
    a list of negative words. The files contain comment lines starting
    with ';' and blank lines, which should be skipped.
    """
    lexicon = []
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon


def load_nyc_names(embeddings, names_path):
    names_df = pd.read_csv(names_path + 'names.csv')
    ethnicity_fixed = []
    for n in names_df['Ethnicity']:
        if n.startswith('BLACK'):
            ethnicity_fixed.append('Black')
        if n.startswith('WHITE'):
            ethnicity_fixed.append('White')
        if n.startswith('ASIAN'):
            ethnicity_fixed.append('Asian')
        if n.startswith('HISPANIC'):
            ethnicity_fixed.append('Hispanic')

    names_df['Ethnicity'] = ethnicity_fixed
    
    names_df = names_df[np.logical_or(names_df['Ethnicity']=='Black', names_df['Ethnicity']=='White')]
    
    names_df['Child\'s First Name'] = [n.lower() for n in names_df['Child\'s First Name']]
    
    names_from_df = names_df['Child\'s First Name'].values.tolist()
    idx_keep = [i for i, n in enumerate(names_from_df) if n in embeddings.index]
    
    names_df = names_df.iloc[idx_keep]
    names_from_df = names_df['Child\'s First Name'].values.tolist()
    all_names_embed = embeddings.loc[names_from_df].values
    
    return all_names_embed, names_from_df


def load_data(data_path, embeddings_path, state=None, names_path=None):
    pos_words = load_lexicon(data_path + '/positive-words.txt')
    neg_words = load_lexicon(data_path + '/negative-words.txt')
    embeddings = load_embeddings(embeddings_path)

    # filter words that do not appear in the embedding index
    pos_words = [word for word in pos_words if word in embeddings.index]
    neg_words = [word for word in neg_words if word in embeddings.index]
    
    pos_vectors = embeddings.loc[pos_words].dropna()
    neg_vectors = embeddings.loc[neg_words].dropna()

    vectors = pd.concat([pos_vectors, neg_vectors])
    targets = np.array([1 for entry in pos_vectors.index] + [-1 for entry in neg_vectors.index])
    labels = list(pos_vectors.index) + list(neg_vectors.index)
    
    if names_path is not None:
        all_names_embed, names_from_df = load_nyc_names(embeddings, names_path)
    else:
        all_names_embed, names_from_df = None, None
        
    if state is None:
        X = vectors.values
        return embeddings, X, targets, labels, all_names_embed, names_from_df
    
    else:
        train_vectors, test_vectors, train_targets, test_targets, train_vocab, test_vocab = \
            train_test_split(vectors, targets, labels, test_size=0.1, random_state=state)
            
        ## Data
        X_train = train_vectors.values
        X_test = test_vectors.values
        
        # Encoding y
        one_hot = OneHotEncoder(sparse=False, categories='auto')
        one_hot.fit(np.array(train_targets).reshape(-1,1))
        y_train = one_hot.transform(np.array(train_targets).reshape(-1,1))
        y_test = one_hot.transform(np.array(test_targets).reshape(-1,1))
        
        return embeddings, X_train, X_test, y_train, y_test, train_vocab, test_vocab, all_names_embed, names_from_df
