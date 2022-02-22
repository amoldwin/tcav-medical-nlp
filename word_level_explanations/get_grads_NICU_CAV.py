import pickle

from dataset import NewsDataset
from BertModel import WeightedBertForSequenceClassification
from BertModel import Bert_cutted_cosine


from smooth_gradient import SmoothGradient
from integrated_gradient import IntegratedGradient

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertTokenizer
import pandas as pd
from IPython.display import display, HTML
import numpy as np
from transformers import BertConfig, BertTokenizer
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

batch_size=1
criterion = nn.CrossEntropyLoss()

config = BertConfig()

# cavs_dir='../torch_tcav/tcav_scripts/tcav_class_test_halfsplitFINALWeightedBlueBert512NicuLosNoneRemovedHalfValTest_1_checkpoint-2375_layer11/cavs/'
# f1s_dct={}
# for fn in os.listdir(cavs_dir):
#     with open(os.path.join(cavs_dir,fn),'rb') as file:
#         cav_dct = pickle.load(file)
#     f1=float(cav_dct['metrics'].split()[12])
#     f1s_dct[fn]=f1
fn='positive_7742-negative500_7742_22-11-linear-0.1.pkl'
fn='positive_V3401-negative500_V3401_40-11-linear-0.1.pkl'
with open('../torch_tcav/tcav_scripts/tcav_class_test_halfsplitFINALWeightedBlueBert512NicuLosNoneRemovedHalfValTest_1_checkpoint-2375_layer11/cavs/'+fn,'rb') as file:
    cav_dct = pickle.load(file)

cav = cav_dct['cavs'][0]

print(cav_dct['metrics'])

model_path = '../torch_tcav/model_train_scripts/checkpoints/FINALWeightedBlueBert512NicuLosNoneRemovedHalfValTest_1/checkpoint-2375'
bert = WeightedBertForSequenceClassification.from_pretrained(model_path,output_attentions=True,cav=cav,layer_number=-1)

bert.classifier.to(bert.bert.device)
bert.classifier.cav_tensor=bert.classifier.cav_tensor.to(bert.bert.device)




train_df = pd.read_csv('../tcav2/data/halfvaltest_los_nicu_train_noneremoved.csv')
    
val_df = pd.read_csv('../tcav2/data/halfvaltest_los_nicu_val_noneremoved.csv')

test_df = pd.read_csv('../tcav2/data/halfvaltest_los_nicu_test_noneremoved.csv')

texts = list(train_df['TEXT'].apply(lambda x: x))+list(val_df['TEXT'].apply(lambda x: x))+list(test_df['TEXT'].apply(lambda x: x))

all_instances = []
# texts=['Patient was a pleasant male in his early 40s who was given plavix and aspirin for CAD and required cabg', 'Patient is healthy']
for i,text in enumerate(texts):
    print(i,flush=True)
    test_example = [
        [text], 
        [""]
    ]

    test_dataset = NewsDataset(
        data_list=test_example,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings, 
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    integrated_grad = IntegratedGradient(
        bert, 
        criterion, 
        tokenizer, 
        show_progress=True,
        encoder="bert"
    )
    instances = integrated_grad.saliency_interpret(test_dataloader)
    all_instances.append(instances)

# coloder_string = integrated_grad.colorize(instances[0])
# display(HTML(coloder_string))

pd.DataFrame([instance[0] for instance in all_instances]).to_csv('all_nicu_grads_CAV_'+fn+'.csv')