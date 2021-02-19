import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

train = pd.read_csv('../Data/train.csv')
acc = pd.DataFrame(columns = ['model', 'method', 'accuracy'])

# clean text
def clean(text):
    nourl = re.sub(r'[\S|\s]http\S+', '', text)
    noat = re.sub(r'\s@\S+', '', nourl)
    tokens = word_tokenize(noat)
    lemmatizer = WordNetLemmatizer()
    lemma = []
    for word in tokens:
        if word not in stopwords.words('english'):
            lemma.append(lemmatizer.lemmatize(word))
    lemma = ' '.join([word for word in lemma])
    nopunct = "".join([char for char in lemma if char not in string.punctuation])
    clean = re.sub(' +', ' ', nopunct)
    
    return clean.lower()

# helpful functions
def append_df(model, method, predictions):
    return acc.append({'model' : model,
                       'method' : method,
                       'accuracy' : accuracy_score(test2['target'], predictions)},
                      ignore_index = True)

train['clean_text'] = train['text'].apply(lambda x : clean(x))

train2 = train.sample(frac=.6, random_state=400)
test2 = train.drop(train2.index)

## TfIdf
tfidf = TfidfVectorizer()

train_tf_features = tfidf.fit_transform(train2['clean_text'])
test_tf_features = tfidf.transform(test2['clean_text'])

# svm
svm_tf_fit = LinearSVC().fit(train_tf_features, train2['target'])
svm_tf_pred = svm_tf_fit.predict(test_tf_features)

acc = append_df('SVM', 'TfIdf', svm_tf_pred)

# logistic regression
log_tf_fit = LogisticRegression().fit(train_tf_features, train2['target'])
log_tf_pred = log_tf_fit.predict(test_tf_features)

acc = append_df('Logistic Regression', 'TfIdf', log_tf_pred)

# random forest
rf_tf_fit = RandomForestClassifier(random_state=400).fit(train_tf_features, train2['target'])
rf_tf_pred = rf_tf_fit.predict(test_tf_features)

acc = append_df('Random Forest', 'TfIdf', rf_tf_pred)

# naive bayes
nb_tf_fit = MultinomialNB().fit(train_tf_features, train2.target)
nb_tf_pred = nb_tf_fit.predict(test_tf_features)

acc = append_df('Naive Bayes', 'TfIdf', nb_tf_pred)

## BoW
bow = CountVectorizer()

train_bow_features = bow.fit_transform(train2['clean_text'])
test_bow_features = bow.transform(test2['clean_text'])

# logistic regression
log_bow_fit = LogisticRegression().fit(train_bow_features, train2['target'])
log_bow_pred = log_bow_fit.predict(test_bow_features)

acc = append_df('Logistic Regression', 'BoW', log_bow_pred)

# svm
svm_bow_fit = LinearSVC().fit(train_bow_features, train2.target)
svm_bow_pred = svm_bow_fit.predict(test_bow_features)

acc = append_df('SVM', 'BoW', svm_bow_pred)

# random forest
rf_bow_fit = RandomForestClassifier(random_state=400).fit(train_bow_features, train2.target)
rf_bow_pred = rf_bow_fit.predict(test_bow_features)

acc = append_df('Random Forest', 'BoW', rf_bow_pred)

# naive bayes
nb_bow_fit = MultinomialNB().fit(train_bow_features, train2.target)
nb_bow_pred = nb_bow_fit.predict(test_tf_features)

acc = append_df('Naive Bayes', 'BoW', nb_bow_pred)
