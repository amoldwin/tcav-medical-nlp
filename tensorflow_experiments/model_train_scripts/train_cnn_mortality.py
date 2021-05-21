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
from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix


train = pd.read_csv('../data/mortality_train.csv')
val = pd.read_csv('../data/mortality_val.csv')

sentences_train = train['TEXT']
y_train = train['label']

sentences_val = val['TEXT']
y_val = val['label']

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(sentences_train)

print('training tokenizer')
X_train = tokenizer.texts_to_sequences(sentences_train)
X_val = tokenizer.texts_to_sequences(sentences_val)
print('tokenizer trained')
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

# round(3.5)

# checkpoint = Checkpoint((X_val,y_val), 'precison.h5')

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self,val_data, y_true, patience=0):
        super(MetricsCallback, self).__init__()
        self.y_true = y_true
        self.val_data = val_data
        # self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('checkpoints/cnn/cnn_mortality'+str(epoch)+'.ckpt')
        y_pred = self.model.predict(self.val_data)
        logits = y_pred
        # Here we get the actual classes
        y_pred = np.round(y_pred)
        # Actual dictionary
        report_dictionary = classification_report(self.y_true, y_pred, output_dict = True)
        # Only printing the report
#         print(classification_report(self.y_true,y_pred,output_dict=False))
        scores_df = pd.DataFrame.from_dict(report_dictionary)
    #         val_loss = loss(self.y_true,logits).numpy()
        auc = roc_auc_score(self.y_true,logits, multi_class='ovr', average='macro')
        print(auc)
        scores_df = scores_df.append({'macro_auc':auc},ignore_index=True)
        scores_df.to_csv('checkpoints/cnn/cnn_mortality_'+str(epoch)+'scores.csv')
metrics_callback = MetricsCallback(val_data =X_val, y_true = y_val)

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
print('starting training')
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_val, y_val),
                    batch_size=10,
              callbacks=[metrics_callback])


# model.save('checkpoints/cnn/cnn_mortality.ckpt')
import pickle

with open('checkpoints/cnn/tokenizer_mortality_cnn.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('tokenizer_final.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# [layer.name for layer in model.layers]

