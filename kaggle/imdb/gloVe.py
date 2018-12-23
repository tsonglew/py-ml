import os

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

imdb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
train_dir = os.path.join(imdb_dir, 'train')


def parse_data(data_dir):
    """
    process IMDB original data and labels
    """
    texts = list()
    labels = list()
    for label in ['neg', 'pos']:
        dir_name = os.path.join(data_dir, label)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname)) as f:
                    texts.append(f.read())
                    labels.append(int(label == 'pos'))
    return texts, labels

texts, labels = parse_data(train_dir)
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000    
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
    
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print(f'Shape of data tensor: {data.shape}; Shape of labels tensor: {labels.shape}')

# shuffle pos and neg samples
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# GloVe trained with Wikipedia 2014: https://nlp.stanford.edu/projects/glove/
glove_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glove')

embeddings_index = dict()
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values =  line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential() 
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(x_train, y_train, 32, 10, validation_data=(x_val, y_val))
model.save_weights(os.path.join(glove_dir, 'pre_trained_glove_model.h5'))