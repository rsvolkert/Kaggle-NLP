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
def fit(model, method):
    if method == 'tfidf':
        train = train_tf_features
        test = test_tf_features
    else:
        train = train_bow_features
        test = test_bow_features
    
    return model.fit(train, train2.target).predict(test)

def append_df(model_name, method, model):
    predictions = fit(model, method)
    return acc.append({'model' : model_name,
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
acc = append_df('SVM', 'tfdif', LinearSVC(random_state=400))

# logistic regression
acc = append_df('Logistic Regression', 'tfdif', LogisticRegression(random_state=400))

# random forest
acc = append_df('Random Forest', 'tfdif', RandomForestClassifier(random_state=400))

# naive bayes
acc = append_df('Naive Bayes', 'tfdif', MultinomialNB())

## BoW
bow = CountVectorizer()

train_bow_features = bow.fit_transform(train2['clean_text'])
test_bow_features = bow.transform(test2['clean_text'])

# svm
acc = append_df('SVM', 'bow', LinearSVC(random_state=400))

# logistic regression
acc = append_df('Logistic Regression', 'bow', LogisticRegression(random_state=400))

# random forest
acc = append_df('Random Forest', 'bow', RandomForestClassifier(random_state=400))

# naive bayes
acc = append_df('Naive Bayes', 'bow', MultinomialNB())
