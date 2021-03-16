import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('../Data/train.csv')

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

## TfIdf
tfidf = TfidfVectorizer(ngram_range = (1,2))

train_tf = tfidf.fit_transform(train.clean_text)

# grid CV
grid = {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'dual' : [True, False],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'C' : [100, 10, 1, 0.1, 0.01]}

model = LogisticRegression(random_state=400)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(train_tf, train.target)

print('Best {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
###### not better than the base logit