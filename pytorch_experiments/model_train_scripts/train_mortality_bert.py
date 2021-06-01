

from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer
import torch
from modeling_bert import WeightedBertForSequenceClassification


from datasets import load_dataset, load_metric
from sklearn.metrics import roc_auc_score, classification_report,average_precision_score, matthews_corrcoef, confusion_matrix

EXP_NAME = 'WeightedBlueBert512DischargeMortalityAttnDropout0.5HddnDrp0.5'

tokenizer=BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

task_name='discharge_mortality'

def preprocess_function(examples):
    examples['labels'] = examples['discharge_mortality']
    examples = tokenizer(examples["text"], truncation=True, max_length=512,padding=True)
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
print('done loading data',flush=True)
def compute_metrics(eval_pred):
#     assert False

    predictions, labels = eval_pred
    print(labels)
    print(predictions)
    class_preds = predictions[:,1]
    class_labels= [[1,0] if x==0 else [0,1] for x in labels]
    auc = roc_auc_score(class_labels,predictions)
    prc = average_precision_score(class_labels,predictions)

    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    mcc = matthews_corrcoef(labels,predictions)
    metrics_dict = classification_report(labels,predictions,output_dict=True)['1']
    print(metrics_dict)
#     metrics_dict = {k:[v] for k,v in metrics_dict.items()}
    metrics_dict['AUC']=auc
    metrics_dict['AUPRC']=prc
    metrics_dict['MCC']=mcc
    
    
    df_dict = {k:[v] for k,v in metrics_dict.items()}
    df_dict['confusion'] = str(confusion_matrix(labels,predictions))
    pd.DataFrame(df_dict).to_csv('checkpoints/'+EXP_NAME+'.csv', mode='a',header=True)
    return metrics_dict

training_args = TrainingArguments(
    output_dir='./checkpoints/'+EXP_NAME,          # output directory
    num_train_epochs=500,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
    logging_steps=1,
    learning_rate=1e-5,
    save_strategy="epoch",
    do_eval=True,
    evaluation_strategy="epoch",
)

weight=torch.FloatTensor([1,len(train_dataset)/sum([x['labels'] for x in train_dataset])]).cuda()
model = WeightedBertForSequenceClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", weight_list=weight, num_labels=2,attention_probs_dropout_prob=0.5,
        hidden_dropout_prob=0.5)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,    # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

