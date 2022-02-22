import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from BertModel import WeightedBertForSequenceClassification


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


model_path = '../torch_tcav/model_train_scripts/checkpoints/FINALWeightedBlueBert512NicuLosAdmissions_2/checkpoint-26015'
bert = WeightedBertForSequenceClassification.from_pretrained(model_path,output_attentions=True)

test_df = pd.read_csv('../tcav2/data/los_nicu_test_admissions.csv')


test_df

rows = []
for row in test_df.iterrows():
    txt=row[1]['TEXT']
    label=row[1]['label']

    tokens = torch.LongTensor([tokenizer.encode(txt, truncation=True, max_length=512)])

    out = bert(input_ids = tokens, return_logits=False, return_dict=True)

    prob = torch.softmax(out['logits'], 1 ).detach().numpy()[0][1]

    attentions = out['attentions']

    attns = np.array([x.detach().numpy()[0] for x in attentions]).mean(0).mean(0).mean(0)
    rows.append([txt,label,tokens,prob,attns])

nicu_attentions_df = pd.DataFrame(rows, columns=['TEXT','label', 'ids','prob','attention'])

nicu_attentions_df['tokens']=nicu_attentions_df['TEXT'].apply(lambda x: list(tokenizer.encode(x, truncation=True, max_length=512)))

nicu_attentions_df['attention'] = nicu_attentions_df['attention'].apply(lambda x: list(x) )

nicu_attentions_df['ids']=nicu_attentions_df['ids'].apply(lambda x: [int(y.detach()) for y in x[0]])

nicu_attentions_df.to_csv('nicu_attentions.csv')