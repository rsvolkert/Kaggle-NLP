import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../Data/train.csv')
acc = pd.DataFrame(columns = ['model', 'accuracy'])

# clean text
def clean(text):
    nourl = re.sub(r'[\S|\s]http\S+', '', text)
    noat = re.sub(r'\s@\S+', '', nourl)
    tokens = word_tokenize(noat)
    nostop = ' '.join([word for word in tokens if word not in stopwords.words('english')])
    nopunct = "".join([char for char in nostop if char not in string.punctuation])
    clean = re.sub(' +', ' ', nopunct)
    
    return clean.lower()

# helpful functions
def append_df(model, predictions):
    return acc.append({'model' : model,
                       'accuracy' : accuracy_score(test2['target'], predictions)},
                      ignore_index = True)

train['clean_text'] = train['text'].apply(lambda x : clean(x))

train2 = train.sample(frac=.6, random_state=400)
test2 = train.drop(train2.index)

# vectorize
tfidf = TfidfVectorizer()

train_features = tfidf.fit_transform(train2['clean_text'])
test_features = tfidf.transform(test2['clean_text'])

# svm
svm_fit = LinearSVC().fit(train_features, train2['target'])
svm_pred = svm_fit.predict(test_features)

acc = append_df('SVM', svm_pred)

# logistic regression
log_fit = LogisticRegression().fit(train_features, train2['target'])
log_pred = log_fit.predict(test_features)

acc = append_df('Logistic Regression', log_pred)

# random forest
rf_fit = RandomForestClassifier(random_state=400).fit(train_features, train2['target'])
rf_pred = rf_fit.predict(test_features)

acc = append_df('Random Forest', rf_pred)
