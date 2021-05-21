import re
from nltk import TreebankWordTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import DefaultFlowCallback, PrinterCallback 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np
import os
import random
import time

import logging
logging.basicConfig(level=logging.ERROR)
from transformers import TFBertPreTrainedModel,TFDistilBertMainLayer
from transformers.modeling_tf_utils import (
    TFQuestionAnsweringLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    shape_list,
)
import pandas as pd
from sklearn.metrics import roc_auc_score


from tensorflow import keras

print('loading training and val data')
print('loading data', flush=True)
time.sleep(.2)
source_dir = './'



def convert_example_to_feature(review):
  
  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
    return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True,
                return_token_type_ids = True
              )
# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    
    for (i, row) in enumerate(ds.values):
#     for index, row in ds.iterrows():
#         review = row["text"]
#         label = row["y"]
        review = row[0]
        label = list(row[1:])
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

   
class TFBertForMultilabelClassification(TFBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultilabelClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert =  TFDistilBertMainLayer(config, name="distilbert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='sigmoid')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        
        pooled_output = outputs[0][:,0]

        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)



def measure_auc(label,pred):
  auc = [roc_auc_score(label[:,i],pred[:,i]) for i in list(range(6))]
  return pd.DataFrame({"label_name":["toxic","severe_toxic","obscene","threat","insult","identity_hate"],"auc":auc})

 # parameters

 
print('loading data', flush=True)
print('loading data')

val_data =  pd.read_csv('../data/SAPSII_val.csv')
val_data=val_data[[val_data.columns[2]]+list(val_data.columns[3:])]

model_path = "distilbert-base-uncased" #模型路径，建议预先下载(https://huggingface.co/bert-base-uncased#)

# parameters
max_length = 512
batch_size = 9
learning_rate = 2e-5
number_of_epochs = 100
num_classes = len(val_data.iloc[0])-1

# tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# val dataset
ds_val_encoded = encode_examples(val_data)
# test dataset
# ds_test_encoded = encode_examples(test_data).batch(batch_size)
epoch=8
saved_path = 'checkpoints/distilbert_sapsii__'+str(epoch)+'.h5'
model = TFBertForMultilabelClassification.from_pretrained(saved_path)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()

print('compiling', flush=True)
print('loading and compiling model')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report 

class MetricsCallback(Callback):
    def __init__(self, val_data, y_true,model):
        # Should be the label encoding of your classes
        self.y_true = y_true
        self.val_data = val_data
        
    # def on_epoch_end(self, logs):
        # Here we get the probabilities
        y_pred = model.predict(self.val_data)[0]#['logits']
        logits = y_pred
        # Here we get the actual classes
        y_pred = np.round(y_pred)
        # Actual dictionary
        report_dictionary = classification_report(self.y_true, y_pred, output_dict = True)
        # Only printing the report
#         print(classification_report(self.y_true,y_pred,output_dict=False))
        scores_df = pd.DataFrame.from_dict(report_dictionary)
        val_loss = loss(self.y_true,logits).numpy()
        try:
            auc = roc_auc_score(self.y_true,logits, multi_class='ovr', average='macro')
        except:
            auc=0
        print(auc)
        scores_df = scores_df.append({'val_loss':val_loss,'macro_auc':auc},ignore_index=True)
        scores_df.to_csv(source_dir+'checkpoints/distilbert_sapsii_'+str(epoch)+'scores.csv')
        

metrics_callback = MetricsCallback(val_data = np.array([x[0]['input_ids'] for x in list(ds_val_encoded)]), y_true = [x[1] for x in list(ds_val_encoded)], model=model)
print('fitting', flush=True)
print('fitting model')

# fit model
batch_size=8
bert_history = model.evaluate(callbacks=[metrics_callback])

# evaluate val_set
pred=model.predict(ds_val_encoded)[0]
df_auc = measure_auc(val_data.iloc[:,2:].astype(np.float32).values,pred)
print("val set mean column auc:",df_auc)
#predict test_set

