import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
train = pd.read_csv('Data/train.csv')


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


train['clean_text'] = train['text'].apply(lambda x: clean(x))

# TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2))

train_tf = tfidf.fit_transform(train.clean_text)

# Logistic Regression

model = LogisticRegression(random_state=400)
model.fit(train_tf, train['target'].values)
test = pd.read_csv('Data/test.csv')
test['clean_text'] = test['text'].apply(lambda x: clean(x))
test_tf = tfidf.transform(test['clean_text'])
ypred = model.predict(test_tf)
test['target'] = ypred
submission = test.drop(['keyword', 'location', 'text', 'clean_text'], axis=1)
submission.to_csv('Data/submission.csv', index=False)
