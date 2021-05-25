import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import tensorflow.keras.backend as K
from sklearn.metrics import recall_score, precision_score

train = pd.read_csv('../tcav2/data/CAD_train.csv')
val = pd.read_csv('../tcav2/data/CAD_val.csv')


sentences_train = train['TEXT']
y_train = train['label']

sentences_val = val['TEXT']
y_val = val['label']

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(sentences_val)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_val = tokenizer.texts_to_sequences(sentences_val)

vocab_size = len(tokenizer.word_index) + 1

maxlen=4096
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)


class Checkpoint(keras.callbacks.Callback):

    def __init__(self, test_data, filename):
        self.test_data = test_data
        self.filename = filename

    def on_train_begin(self, logs=None):
        self.pre = [0.]
        self.rec = [0.]
        print('Test on %s begins' % self.filename)

    def on_train_end(self, logs={}):
        print('Best Precison: %s' % max(self.pre))
        print('Best Recall: %s' % max(self.rec))
        return

    def on_epoch_end(self, epoch, logs={}):
        
        x, y = self.test_data
        preds = self.model.predict(x)
#         print(preds.shape, precision(preds, y) )
        prec = precision_score([round(i[0]) for i in preds], y)
        rec = recall_score([round(i[0]) for i in preds], y)
        self.pre.append(prec)
        self.rec.append(rec)
        print(' \nprecision: ' + str(prec)+' recall:'+str(rec))

        # print your precision or recall as you want
        print(...)

        # Save your model when a better trained model was found
#         if pre > max(self.pre):
#             self.model.save(self.filename, overwrite=True)
#             print('Higher precision found. Save as %s' % self.filename)
        return

round(3.5)

checkpoint = Checkpoint((X_val,y_val), 'precison.h5')

class save_every_epoch(keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(save_every_epoch, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('cnn__all_epochs/epoch_'+str(epoch)+'.ckpt')


embedding_dim=50

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy', 
                   metrics = ['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_val, y_val),
                    batch_size=10,
              callbacks=[checkpoint, save_every_epoch()])


model.save('cnn/cnn_final.ckpt')
import pickle

with open('tokenizer_final.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer_final.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

[layer.name for layer in model.layers]

