import torch
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_context(rc={'figure.figsize': (9, 9)}, font_scale=2.)


TOKEN_RE = re.compile(r"\w.*?\b")


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


def load_lexicon(filepath):
    """
    load a file from Bing Liu's sentiment lexicon containing
    English words in Latin-1 encoding

    One file contains a list of positive words, and the other
    contains a list of negative words. The files contain comment
    lines starting with ';' and blank lines, which should be skipped
    """

    lexicon = []
    with open(filepath, encoding='latin-1') as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon


def load_data(data_path, embeddings_path, state=0):

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
    
    train_vectors, test_vectors, train_targets, test_targets, train_vocab, test_vocab = \
        train_test_split(vectors, targets, labels, test_size=0.1, random_state=state)
        
    ## Data
    X_train = train_vectors.values
    X_test = test_vectors.values

    y_train = train_targets
    y_train[y_train == -1] = 0
    y_test = test_targets
    y_test[y_test == -1] = 0
    
    return embeddings, X_train, X_test, y_train, y_test, train_vocab, test_vocab


def load_test_names(embeddings):
    NAMES_BY_ETHNICITY = {
    # The first two lists are from the Caliskan et al. appendix describing the
    # Word Embedding Association Test.
    'White': [
        'Adam', 'Chip', 'Harry', 'Josh', 'Roger', 'Alan', 'Frank', 'Ian', 'Justin',
        'Ryan', 'Andrew', 'Fred', 'Jack', 'Matthew', 'Stephen', 'Brad', 'Greg', 'Jed',
        'Paul', 'Todd', 'Brandon', 'Hank', 'Jonathan', 'Peter', 'Wilbur', 'Amanda',
        'Courtney', 'Heather', 'Melanie', 'Sara', 'Amber', 'Crystal', 'Katie',
        'Meredith', 'Shannon', 'Betsy', 'Donna', 'Kristin', 'Nancy', 'Stephanie',
        'Bobbie-Sue', 'Ellen', 'Lauren', 'Peggy', 'Sue-Ellen', 'Colleen', 'Emily',
        'Megan', 'Rachel', 'Wendy'
    ],

    'Black': [
        'Alonzo', 'Jamel', 'Lerone', 'Percell', 'Theo', 'Alphonse', 'Jerome',
        'Leroy', 'Rasaan', 'Torrance', 'Darnell', 'Lamar', 'Lionel', 'Rashaun',
        'Tyree', 'Deion', 'Lamont', 'Malik', 'Terrence', 'Tyrone', 'Everol',
        'Lavon', 'Marcellus', 'Terryl', 'Wardell', 'Aiesha', 'Lashelle', 'Nichelle',
        'Shereen', 'Temeka', 'Ebony', 'Latisha', 'Shaniqua', 'Tameisha', 'Teretha',
        'Jasmine', 'Latonya', 'Shanise', 'Tanisha', 'Tia', 'Lakisha', 'Latoya',
        'Sharise', 'Tashika', 'Yolanda', 'Lashandra', 'Malika', 'Shavonn',
        'Tawanda', 'Yvette'
    ]
}
    
    NAMES_BY_ETHNICITY['White'] = [n.lower() for n in NAMES_BY_ETHNICITY['White'] if n.lower() in embeddings.index]
    NAMES_BY_ETHNICITY['Black'] = [n.lower() for n in NAMES_BY_ETHNICITY['Black'] if n.lower() in embeddings.index]
    
    white_female_start = NAMES_BY_ETHNICITY['White'].index('amanda')
    black_female_start = NAMES_BY_ETHNICITY['Black'].index('aiesha')
    

    test_gender = white_female_start*['Male'] + (len(NAMES_BY_ETHNICITY['White']) - white_female_start)*['Female']
    test_gender += black_female_start*['Male'] + (len(NAMES_BY_ETHNICITY['Black']) - black_female_start)*['Female']
    test_df = pd.DataFrame({'name':NAMES_BY_ETHNICITY['White'] + NAMES_BY_ETHNICITY['Black'],
                            'race':len(NAMES_BY_ETHNICITY['White'])*['White'] + len(NAMES_BY_ETHNICITY['Black'])*['Black'],
                            'gender':test_gender})
    
    test_names_embed = embeddings.loc[test_df['name']].values
    
    return test_df, test_names_embed

def load_nyc_names(names_path, embeddings):
    names_df = pd.read_csv(names_path)

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
    idx_keep = []
    for i, n in enumerate(names_from_df):
        if n in embeddings.index:
            idx_keep.append(i)
    
    names_df = names_df.iloc[idx_keep]
    names_from_df = names_df['Child\'s First Name'].values.tolist()
    names_embed = embeddings.loc[names_from_df].values
    
    return names_embed


def print_summary(test_df, method_name, test_accuracy):
    
    print(method_name + ' test accuracy %f' % test_accuracy)
    
    mean_sentiments_race = []
    for r in ['Black', 'White']:
        mean_sent = test_df[method_name + '_logits'][test_df['race']==r].mean()
        mean_sentiments_race.append(mean_sent)
        print(method_name + ' %s mean sentiment is %f' %(r, mean_sent))
    print(method_name + ' race mean sentiment difference is %f\n' % np.abs(mean_sentiments_race[0] - mean_sentiments_race[1]))
    
    mean_sentiments_gender = []
    for g in ['Female', 'Male']:
        mean_sent = test_df[method_name + '_logits'][test_df['gender']==g].mean()
        mean_sentiments_gender.append(mean_sent)
        print(method_name + ' %s mean sentiment is %f' %(g, mean_sent))
    print(method_name + ' gender mean sentiment difference is %f\n' % np.abs(mean_sentiments_gender[0] - mean_sentiments_gender[1]))
    
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6*2, 6))
    
    sns.boxplot(x='race', y=method_name + '_logits', data=test_df, ax=axs[0]).set_title(method_name, fontsize=20)
    sns.boxplot(x='gender', y=method_name + '_logits', data=test_df, ax=axs[1]).set_title(method_name, fontsize=20)

    axs[0].set_ylim([-0.1, 1.1])
    axs[0].set_xlabel('Race', size=18)
    axs[0].set_ylabel('Sentiment', size=18, labelpad=-5)

    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel('Gender', size=18)
    axs[1].set_ylabel('Sentiment', size=18, labelpad=-5)

    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.show()

    return


def text_to_sentiment(text, network, embedding, device):
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    
    with torch.no_grad():
        
        sentence_embeddings = []
        for token in tokens:
            vec = embedding.loc[token].dropna()
            sentence_embeddings.append(torch.Tensor(vec).view(1, -1))
            
        sentence_embeddings = torch.cat(sentence_embeddings, dim=0).mean(dim=0, keepdim=True).to(device)
        
        sentiment = network(sentence_embeddings)
        sentiment = torch.nn.functional.softmax(sentiment.mean(dim=0, keepdim=True), dim=-1)
        mean_sentiment = sentiment.data.detach().cpu().numpy()[0]

    return mean_sentiment


def format_sentiment_score(score):

    if score[0] > score[1]:
        return 'Negative with score ' + '{:.2f}%'.format(score[1]*100)
    elif score[1] > score[0]:
        return 'Positive with score ' + '{:.2f}%'.format(score[1]*100)
    return 'Neutral with score ' + '{:.2f}%'.format(score[1]*100)