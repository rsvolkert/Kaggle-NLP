import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import spacy


def clean_text(text):
    """Convert text to lowercase, remove special characters and extra whitespace."""
    text = text.lower()
    # for now, remove hashtags and mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # let dashes and underscores separate words
    text = re.sub(r'[-_]', ' ', text)
    # remove any special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # remove any extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def create_bow_matrix(text_data, min_occur=1, max_occur=None):
    """Convert a series of text data to a BOW matrix."""
    word_freq = text_data \
        .str.split(expand=True) \
        .melt(value_name='word')['word'] \
        .value_counts() \
        .sort_values(ascending=False)
    if min_occur:
        word_freq = word_freq[word_freq >= min_occur]
    if max_occur:
        word_freq = word_freq[word_freq <= max_occur]
    bow_matrix = pd.DataFrame(
        0,
        index=text_data.index,
        columns=word_freq.index
    )
    for i, text in text_data.iteritems():
        for word in text.split():
            if word not in bow_matrix.columns:
                continue
            bow_matrix.loc[i, word] += 1
    return bow_matrix


def count_named_entities(text_data, include_entities=None):
    entity_count = pd.Series(0, index=text_data.index, name='named_entities')
    nlp = spacy.load('en_core_web_sm')
    for i, text in text_data.iteritems():
        doc = nlp(text)
        entity_count[i] = sum(1 for e in doc.ents
                              if e.label_ in include_entities)
    return entity_count


def main():
    pd.options.mode.chained_assignment = None
    df = pd.read_csv('../Data/train.csv')
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
    df_train['clean_text'] = df_train['text'].apply(clean_text)
    bow_matrix_train = create_bow_matrix(
        df_train['clean_text'],
        min_occur=2,
        max_occur=200
    )
    bow_matrix_train['n_entities'] = count_named_entities(
        df_train['clean_text'],
        include_entities=['DATE', 'EVENT', 'FAC', 'GPE', 'LOC',
                          'ORG', 'PERSON', 'PRODUCT', 'TIME']
    )
    rf = RandomForestClassifier()
    g = GridSearchCV(
        rf,
        scoring='accuracy',
        cv=5,
        param_grid={'n_estimators': [10, 100, 500],
                    'max_depth': [None, 1, 2, 4, 8],
                    'ccp_alpha': [0.0, 0.1, 0.2, 0.4, 0.8]}
    )
    g.fit(bow_matrix_train.values, df_train['target'].values)
    print(g.best_score_)
    print(g.best_params_)
