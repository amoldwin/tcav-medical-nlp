import pandas as pd
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np
import os
import random
from tensorflow import keras
import time

print('loading data', flush=True)
time.sleep(.2)
train = pd.read_csv('../data/mortality_train.csv')
val = pd.read_csv('../data/mortality_val.csv')


X_train = train['TEXT']
y_train = train['label']

X_val = val['TEXT']
y_val = val['label']
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

auc = tf.keras.metrics.AUC()
def simple_bert():
    
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", seq_classif_dropout=0.5, attention_dropout=0.5,dropout=0.5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss=loss)
    
    return model
print('loading training and val data')

train_encodings = tokenizer(list(X_train), is_split_into_words=False, padding=True, truncation=True)
val_encodings = tokenizer(list(X_val), is_split_into_words=False, padding=True, truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))

tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)
print('loading model', flush=True)
time.sleep(.2)
print('loading and compiling model')
model = simple_bert()

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report 
source_dir = './'
class MetricsCallback(Callback):
    def __init__(self, val_data, y_true):
        # Should be the label encoding of your classes
        self.y_true = y_true
        self.val_data = val_data
            
    def on_epoch_end(self, epoch, logs):
        # Here we get the probabilities
        y_pred = self.model.predict(self.val_data)['logits']
        
        print(y_pred)
        logits = y_pred
        y_pred = tf.argmax(y_pred,axis=1)
        # Here we get the actual classes
#         y_pred = np.round(y_pred)
        # Actual dictionary
        report_dictionary = classification_report(self.y_true, y_pred, output_dict = True)
        # Only printing the report
#         print(classification_report(self.y_true,y_pred,output_dict=False))
        scores_df = pd.DataFrame.from_dict(report_dictionary)
        val_loss = loss(self.y_true,logits).numpy()
        scores_df = scores_df.append({'loss':logs['loss'], 'val_loss':val_loss},ignore_index=True)
        scores_df.to_csv(source_dir+'checkpoints/distilbert_weighted/distilbert_scdropout0.5attdropout0.5drouout0.5_binaryce_mortality_weighted'+str(epoch)+'scores.csv')
        try:
            self.model.save_pretrained(source_dir+'checkpoints/distilbert_weighted/distilbert_scdropout0.5attdropout0.5drouout0.5_binaryce_mortality_weighted'+str(epoch)+'.h5')
        except:
            print('save failed')
            pass
        

metrics_callback = MetricsCallback(val_data = np.array([x[0]['input_ids'] for x in list(val_dataset)]), y_true = np.array([x[1].numpy() for x in list(val_dataset)]))
# for i in range(100):

print('fitting model', flush=True)
time.sleep(.2)
batch_size=8
history = model.fit(train_dataset.shuffle(1000).batch(batch_size), epochs=100,validation_data = val_dataset.shuffle(1000).batch(batch_size),batch_size=batch_size, callbacks=[metrics_callback], class_weight={0:1, 1:(len(y_train)/sum(y_train))}  )
    # pickle.dump(history,open('bertcheckpoints/Bert_unweighted'+str(i)+'.pickle','wb') )
    # model.save_pretrained('bertcheckpoints/distilbert/distilbert_cad_weighted_epoch'+str(i)+'.h5')

