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

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('../data/mortality_train.csv')
val = pd.read_csv('../data/mortality_val.csv')

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

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self,val_data, y_true, patience=0):
        super(MetricsCallback, self).__init__()
        self.y_true = y_true
        self.val_data = val_data
        # self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('checkpoints/cnn/cnn_berttokenize_mortality'+str(epoch)+'.ckpt')
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
        scores_df.to_csv('checkpoints/cnn/cnn_berttokenize_mortality_'+str(epoch)+'scores.csv')
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

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_val, y_val),
                    batch_size=10,
              callbacks=[metrics_callback])


# model.save('cnn/cnn_final.ckpt')
# import pickle

# with open('tokenizer_final.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('tokenizer_final.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# [layer.name for layer in model.layers]

