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
from transformers import DistilBertTokenizer



train = pd.read_csv('../tcav2/data/CAD_train.csv')
val = pd.read_csv('../tcav2/data/CAD_val.csv')


sentences_train = train['TEXT']
y_train = train['label']

sentences_val = val['TEXT']
y_val = val['label']

# train_encodings = tokenizer(list(X_train), is_split_into_words=False, padding=True, truncation=True)
# val_encodings = tokenizer(list(X_val), is_split_into_words=False, padding=True, truncation=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def distilbert_encode(data):
    encoded =  tokenizer.encode_plus(
            text=data,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True,
            truncation=True
            )
    encoded=encoded['input_ids']
    encoded = encoded 
    encoded = encoded + [tokenizer.pad_token_id]*(512-len(encoded))
    return np.asarray(encoded)

vocab_size = tokenizer.vocab_size + 1

maxlen=512
X_train = pad_sequences(list( sentences_train.apply(lambda x: list(distilbert_encode(x))) ), padding='post', maxlen=512  )
X_val = pad_sequences(list( sentences_val.apply(lambda x: list(distilbert_encode(x))) ), padding='post', maxlen=512)


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
        self.model.save('cnn__all_epochs/epoch_'+str(epoch)+'berttokenize.ckpt')


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


# model.save('cnn/cnn_final.ckpt')
# import pickle

# with open('tokenizer_final.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('tokenizer_final.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# [layer.name for layer in model.layers]

