from modeling_bert import WeightedBertForSequenceClassificationWithTwinResidualNetwork
import torch

from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer
import torch
from modeling_bert import WeightedBertForSequenceClassification,WeightedBertForSequenceClassificationWithTwinResidualNetwork
import modeling_bert
from sklearn.metrics import accuracy_score
import shutil
from datasets import load_dataset, load_metric, concatenate_datasets
from sklearn.metrics import roc_auc_score, classification_report,average_precision_score, matthews_corrcoef, confusion_matrix
import wandb



dirs_dct = dict(list(pd.read_csv('../directory_paths.csv')['paths'].apply(eval)))
checkpoints_dir = dirs_dct['checkpoints_dir']

EXP_NAME = 'residual_CCS_'

wandb.init(settings=wandb.Settings(start_method="fork"))

for i in range(100):
    if EXP_NAME +'_'+str(i)+'.csv' not in os.listdir(checkpoints_dir):
        EXP_NAME = EXP_NAME +'_'+str(i)
        break
print('experiment name is '+ EXP_NAME)

tokenizer=BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

task_name='CCS_txt_str'

def preprocess_function(examples):
    ret = {}
    ret = tokenizer(examples["text"], truncation=True, max_length=512,padding=True)
    ret['labels'] = examples['CCS_txt_str']
    ret['structured_features'] =  [eval(x) for x in examples['structured_features']]
    return ret

tokenizer(['Example this is an', 'another ex'])

def load_dataset_split(split):
    dataset = load_dataset('data_loader.py', task_name,
                           data_dir=dirs_dct['data_dir'],
                           split=split)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return encoded_dataset

print('loading data',flush=True)
train_dataset=load_dataset_split("train")
val_dataset=load_dataset_split("validation")
test_dataset=load_dataset_split("test")
val_test_ds = concatenate_datasets([val_dataset, test_dataset])

# val_dataset['structured_features']

# val_structured_features

# val_dataset['input_ids'][1]

print('done loading data',flush=True)
reverse_labels_dict = {'0':'0','1':'1'}
output_dir=os.path.join(checkpoints_dir,EXP_NAME)
epoch=0
best_mcc=0
best_ckpt=''


reverse_labels_dict = {'0':'0','1':'1'}
output_dir=os.path.join(checkpoints_dir,EXP_NAME)

batch_size=8

def compute_metrics(eval_pred,  output_dir = output_dir):
    val_eval_pred = (eval_pred[0][:len(val_dataset)],eval_pred[1][:len(val_dataset)] )
    test_eval_pred = (eval_pred[0][len(val_dataset):],eval_pred[1][len(val_dataset):] )
#     assert False
    epoch=1
    best_mcc=0
    best_ckpt=''
    if EXP_NAME+'.csv' in os.listdir(checkpoints_dir):
        old_scores = pd.read_csv(os.path.join(checkpoints_dir,EXP_NAME+'.csv'))
        best_mcc=old_scores['val_MCC'].max()
        epoch = old_scores['epoch'].iloc[-1]+1
        best_ckpt = old_scores.iloc[old_scores['val_MCC'].idxmax()]['fn']
    metrics_dict = {}
    for name, dset  in {'val_':val_eval_pred, 'test_':test_eval_pred}.items() :
        predictions, labels = dset
        print(name)
#         if name =='val_':
#             print(val_dataset['labels'])
#         else:
#             print(test_dataset['labels'])
#         print(predictions)
#         print(name)
#         print(labels==val_dataset['labels'])
#         print(labels==test_dataset['labels'])
        class_preds = predictions[:,1]
        class_labels= [[1,0] if x==0 else [0,1] for x in labels]
        auc = roc_auc_score(class_labels,predictions)
        prc = average_precision_score(class_labels,predictions)
        auc_1 = roc_auc_score(labels,predictions[:,1])
        prc_1 = average_precision_score(labels,predictions[:,1])

        predictions = np.argmax(predictions, axis=1)
#         print(predictions)
        mcc = matthews_corrcoef(labels,predictions)
        if name=='val_':
            val_mcc=mcc
        acc = accuracy_score(labels, predictions)

        class_report = classification_report(labels,predictions,output_dict=True) 
        metrics_dict[name+'precision'] = class_report['1']['precision']
        metrics_dict[name+'recall'] = class_report['1']['recall']
        metrics_dict[name+'f1-score'] = class_report['1']['f1-score']
        metrics_dict[name+'support'] = class_report['1']['support']
        metrics_dict[name+'Accuracy'] = acc
        metrics_dict[name+'AUC_avg']=auc
        metrics_dict[name+'AUPRC_avg']=prc
        metrics_dict[name+'AUC_1']=auc_1
        metrics_dict[name+'AUPRC_1']=prc_1
        metrics_dict[name+'MCC']=mcc
        metrics_dict[name+'specificity'] = class_report['0']['recall']
    fn='checkpoint-'+str(int(epoch*np.ceil(len(train_dataset)/batch_size) ))
    print(epoch, fn)
    if val_mcc>best_mcc+0.01:
        best_mcc=val_mcc
        best_ckpt = fn
        print('BEST!!!')
    print(os.listdir(output_dir))
    for f in os.listdir(output_dir):
        if not f==best_ckpt:
            shutil.rmtree(output_dir+'/'+f)
    df_dict = {k:[v] for k,v in metrics_dict.items()}
    df_dict['steps'] = epoch*np.ceil(len(train_dataset)/batch_size)
    df_dict['confusion'] = str(confusion_matrix(labels,predictions))
    df_dict['epoch']=epoch
    df_dict['fn']=fn
    if EXP_NAME+'.csv' not in os.listdir(checkpoints_dir):
        pd.DataFrame(df_dict).to_csv(os.path.join(checkpoints_dir, EXP_NAME+'.csv'), mode='a',header=True)
    else:
        pd.DataFrame(df_dict).to_csv(os.path.join(checkpoints_dir, EXP_NAME+'.csv'), mode='a',header=False)
    return metrics_dict

training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    num_train_epochs=100,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_steps=1,
    learning_rate=1e-5,
    save_strategy="epoch",
    do_eval=True,
    evaluation_strategy="epoch",
)

weight=torch.FloatTensor([1,len(train_dataset)/sum([x['labels'] for x in train_dataset])]).cuda()

num_struct = len(train_dataset['structured_features'][0])

model = modeling_bert.WeightedBertForSequenceClassificationWithTwinResidualNetwork.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", weight_list=weight, num_structured_features=num_struct, num_labels=2, pretrained_residual = None )

# model(torch.tensor(train_dataset['input_ids'][0]))


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,    # training dataset
    eval_dataset=val_test_ds,             # evaluation dataset
    compute_metrics=compute_metrics
)
trainer.train()

