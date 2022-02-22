from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import os
import torch
import wandb
# wandb.init(project='medical-nlp')

from datasets import load_dataset, load_metric
from sklearn.metrics import roc_auc_score, classification_report,average_precision_score, matthews_corrcoef, confusion_matrix
from transformers import BertForTokenClassification

EXP_NAME = 'NER_BlueBert'

task_name='NER'

def preprocess_function(examples):
    examples['labels'] = examples['NER']
    examples = {'input_ids':[eval(x) for x in examples['input_ids']],
                'labels':[eval(x) for x in examples['labels']]}
    return examples



def load_dataset_split(split):
    dataset = load_dataset('data_loader.py', task_name,
                           data_dir='../../tcav2/data/',
                           split=split)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return encoded_dataset

print('loading data',flush=True)
train_dataset=load_dataset_split("train")
val_dataset=load_dataset_split("validation")

pd.read_csv('../../tcav2/data/NER_val.csv').iloc[0]

def unravel(lst):
    return [i for j in lst for i in j]

labels_list =['B-CAD', 'B-DIABETES', 'B-FAMILY_HIST', 'B-HYPERLIPIDEMIA', 'B-HYPERTENSION', 'B-MEDICATION', 'B-OBESE', 'B-PHI', 'B-SMOKER', 'I-CAD', 'I-DIABETES', 'I-FAMILY_HIST', 'I-HYPERLIPIDEMIA', 'I-HYPERTENSION', 'I-MEDICATION', 'I-OBESE', 'I-PHI', 'I-SMOKER', 'O']

print(labels_list)

labels_dict = {k:i for i,k in enumerate(labels_list)}
reverse_labels_dict = {str(i):k for i,k in enumerate(labels_list)}


def sigmoid(z):
    return 1/(1 + np.exp(-z))
def compute_metrics(eval_pred):
#     assert False

    predictions, labels = eval_pred
#     print(labels)
#     print(predictions)
#     print(predictions.shape, labels.shape)
#     class_preds = predictions[:,1]
#     class_labels= [[1,0] if x==0 else [0,1] for x in labels]
#     auc = roc_auc_score(class_labels,predictions)
#     prc = average_precision_score(class_labels,predictions)
    labels=unravel(labels)
    sigmoids = sigmoid(predictions)
    predictions = unravel(np.argmax(predictions,axis=2))

#     print('after_argmax',predictions)
#     mcc = matthews_corrcoef(labels,predictions)
    metrics_dict = classification_report(labels,predictions,output_dict=True, labels=list(labels_dict.values()))
    print(metrics_dict)
#     metrics_dict = {k:[v] for k,v in metrics_dict.items()}
#       metrics_dict['AUC']=auc
#     metrics_dict['AUPRC']=prc
#     metrics_dict['MCC']=mcc
    
    
    df_dict = {k:[v] for k,v in metrics_dict.items()}
    df_dict['confusion'] = str(confusion_matrix(labels,predictions))
    pd.DataFrame(df_dict).to_csv('checkpoints/'+EXP_NAME+'.csv', mode='a',header=True)
    reverse_labels_dict['weighted avg'] = 'weighted_avg'
    reverse_labels_dict['macro avg'] = 'macro_avg'
    reverse_labels_dict['micro avg'] = 'micro_avg'
    
    metrics_dict = {reverse_labels_dict[k]+'_'+kk:v[kk] for k,v in metrics_dict.items() if type(v)==dict and k in reverse_labels_dict.keys()  for kk in v.keys()}
    
#     for i in range(len(labels_list)):
#         metrics_dict[reverse_labels_dict[str(i)]]['auc'] = roc_auc_score( labels,unravel(sigmoids[:,:,i]))
        

    return metrics_dict

training_args = TrainingArguments(
    output_dir='./checkpoints/'+EXP_NAME,          # output directory
    num_train_epochs=100,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
    logging_steps=len(train_dataset),
    learning_rate=1e-5,
    save_strategy="epoch",
    do_eval=True,
    evaluation_strategy="epoch",
)
model = BertForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", num_labels=len(labels_list))

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,    # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()