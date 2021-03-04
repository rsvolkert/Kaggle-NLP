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
    clean_text = re.sub(' +', ' ', nopunct)
    
    return clean_text.lower()

train['clean_text'] = train['text'].apply(lambda x : clean(x))

# helpful functions
def append_df(model_name, method, predictions):
    return acc.append({'model' : model_name,
                       'method' : method,
                       'accuracy' : accuracy_score(test2['target'], predictions)},
                      ignore_index = True)


train2 = train.sample(frac=.6, random_state=400)
test2 = train.drop(train2.index)

## TfIdf
tfidf = TfidfVectorizer(ngram_range = (1,2))

train_tf = tfidf.fit_transform(train2['clean_text'])
test_tf = tfidf.transform(test2['clean_text'])

# svm
svm_tfidf = LinearSVC(random_state=400)
svm_tfidf_fit = svm_tfidf.fit(train_tf, train2.target)
svm_tfidf_pred = svm_tfidf_fit.predict(test_tf)
acc = append_df('SVM', 'tfidf', svm_tfidf_pred)

# logistic regression
log_tfidf = LogisticRegression(random_state=400)
log_tfidf_fit = log_tfidf.fit(train_tf, train2.target)
log_tfidf_pred = log_tfidf_fit.predict(test_tf)
acc = append_df('Logistic Regression', 'tfidf', log_tfidf_pred)

# random forest
rf_tfidf = RandomForestClassifier(random_state=400)
rf_tfidf_fit = rf_tfidf.fit(train_tf, train2.target)
rf_tfidf_pred = rf_tfidf_fit.predict(test_tf)
acc = append_df('Random Forest', 'tfidf', rf_tfidf_pred)

# naive bayes
nb_tfidf = MultinomialNB()
nb_tfidf_fit = nb_tfidf.fit(train_tf, train2.target)
nb_tfidf_pred = nb_tfidf_fit.predict(test_tf)
acc = append_df('Naive Bayes', 'tfidf', nb_tfidf_pred)

## BoW
bow = CountVectorizer()

train_bow = bow.fit_transform(train2['clean_text'])
test_bow = bow.transform(test2['clean_text'])

# svm
svm_bow = LinearSVC(random_state=400)
svm_bow_fit = svm_bow.fit(train_bow, train2.target)
svm_bow_pred = svm_bow_fit.predict(test_bow)
acc = append_df('SVM', 'bow', svm_bow_pred)

# logistic regression
log_bow = LogisticRegression(random_state=400)
log_bow_fit = log_bow.fit(train_bow, train2.target)
log_bow_pred = log_bow_fit.predict(test_bow)
acc = append_df('Logistic Regression', 'bow', log_bow_pred)

# random forest
rf_bow = RandomForestClassifier(random_state=400)
rf_bow_fit = rf_bow.fit(train_bow, train2.target)
rf_bow_pred = rf_bow_fit.predict(test_bow)
acc = append_df('Random Forest', 'bow', rf_bow_pred)

# naive bayes
nb_bow = MultinomialNB()
nb_bow_fit = nb_bow.fit(train_bow, train2.target)
nb_bow_pred = nb_bow_fit.predict(test_bow)
acc = append_df('Naive Bayes', 'bow', nb_bow_pred)