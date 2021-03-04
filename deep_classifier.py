import re
import string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

def clean(text):
    nourl = re.sub(r'[\S|\s]http\S+', '', text)
    noat = re.sub(r'\s@\S+', '', nourl)
    nopunct = "".join([char for char in noat if char not in string.punctuation])
    clean_text = re.sub(' +', ' ', nopunct)
    return clean_text.lower()

train = pd.read_csv('../Data/train.csv')
train['clean_text'] = train.text.apply(lambda tweet : clean(tweet))

train2 = train.sample(frac=.6, random_state=400)
test2 = train.drop(train2.index)

# tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.clean_text)

max_len = max([len(tweet.split()) for tweet in train.text])

vocab_size = len(tokenizer.word_index) + 1

train_tokens =  tokenizer.texts_to_sequences(train2.clean_text)
test_tokens = tokenizer.texts_to_sequences(test2.clean_text)

train_pad = pad_sequences(train_tokens, maxlen=max_len, padding='post')
test_pad = pad_sequences(test_tokens, maxlen=max_len, padding='post')

# get pretrained word embeddings
embeddings_dictionary = {}
glove_file = open('../Data/glove.twitter.27B.100d.txt', encoding='utf8')

for line in glove_file:
    records = line.split()
    word = records[0]
    dims = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = dims
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
# model training
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_pad, train2.target, epochs=100, verbose=1)
loss, accuracy = model.evaluate(test_pad, test2.target, verbose=0)
