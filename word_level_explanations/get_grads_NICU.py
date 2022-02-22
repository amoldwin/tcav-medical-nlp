import pickle

from dataset import NewsDataset
from BertModel import WeightedBertForSequenceClassification

from smooth_gradient import SmoothGradient
from integrated_gradient import IntegratedGradient

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from IPython.display import display, HTML
import pandas as pd
torch.cuda.empty_cache() 

config = BertConfig()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# distilbert = DistilBertForSequenceClassification(config, num_labels=2)
model_path = '../torch_tcav/model_train_scripts/checkpoints/FINALWeightedBlueBert512NicuLosAdmissions_2/checkpoint-26015'
bert = WeightedBertForSequenceClassification.from_pretrained(model_path,output_attentions=True)
# distilbert = DistilBertForSequenceClassification(config=config)

criterion = nn.CrossEntropyLoss()

batch_size = 1

# path = './distilbert_cad_final1.pt'

# if torch.cuda.is_available():
#     distilbert.load_state_dict(
#         torch.load(path)
#     )
# else:
#     distilbert.load_state_dict(
#         torch.load(path, map_location=torch.device('cpu'))
#     )
    
train_df = pd.read_csv('../tcav2/data/los_nicu_train_admissions.csv')
    
val_df = pd.read_csv('../tcav2/data/los_nicu_val_admissions.csv')

test_df = pd.read_csv('../tcav2/data/los_nicu_test_admissions.csv')

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

pd.DataFrame([instance[0] for instance in all_instances]).to_csv('all_nicu_grads.csv')