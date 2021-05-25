import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize.punkt import PunktSentenceTokenizer as pst
from transformers import BertTokenizerFast, BatchEncoding
# PST = pst()
import warnings
from tensorflow.keras.callbacks import *
import random
from brat_parser import get_entities_relations_attributes_groups


tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
val_filenames = ['259-04.xml', '259-03.xml', '357-02.xml', '185-01.xml', '156-01.xml', '337-03.xml', '223-03.xml', '361-02.xml', '128-04.xml', '351-02.xml', '397-02.xml', '121-01.xml', '397-03.xml', '149-02.xml', '329-03.xml', '174-01.xml', '360-03.xml', '128-01.xml', '145-01.xml', '368-03.xml', '282-04.xml', '176-04.xml', '326-03.xml', '148-02.xml', '103-03.xml', '157-03.xml', '304-03.xml', '395-03.xml', '322-05.xml', '333-01.xml']

def get_complete_annotations():
    # train_complete_annotations_paths = ['../../n2c2/training-RiskFactors-Complete-Set1/','../../n2c2/training-RiskFactors-Complete-Set2/']
    train_complete_annotations_paths = ['../../n2c2/complete/', '../../n2c2/testing-RiskFactors-Complete/']
    rows = []
    childnum = 0
    gold_rows = []
    for train_complete_annotations_path in train_complete_annotations_paths:
        for fn in os.listdir(train_complete_annotations_path):
            tree = ET.parse(train_complete_annotations_path+'/'+fn)
            root = tree.getroot()
            (TEXT,TAGS) = [child for child in root]
            notetext = TEXT.text
            for child in TAGS:
#                 print(child.tag)
                gold_row = child.attrib
                gold_row['filename'] = fn
                gold_row['note_text'] = notetext
                gold_row['tag'] = child.tag
                gold_row['observation_number'] = childnum
                gold_row['evidence_spans'] = []
                gold_row['text_cues'] = []
                row = child.attrib
                if 'start' in row.keys():
                    gold_row['evidence_spans'].append((int(row['start']),int(row['end'])))
                    gold_row['text_cues'].append(row['text'])
                for grandchild in child:
                    row = grandchild.attrib
                    row['filename'] = fn
                    row['note_text'] = notetext

#                     row['text'] = notetext
                    row['tag'] = child.tag
                    row['observation_number'] = childnum
                    rows.append(row)
                    if 'start' in row.keys():
                        gold_row['evidence_spans'].append((int(row['start']),int(row['end'])))
                        gold_row['text_cues'].append(row['text'])
                gold_rows.append(gold_row)
                childnum+=1
    return pd.DataFrame(rows), pd.DataFrame(gold_rows)


def get_gold_annotations():
    train_gold_annotations_paths = ['../../n2c2/training-RiskFactors-Gold-Set1/','../../n2c2/training-RiskFactors-Gold-Set2/']
    rows = []
    for train_gold_annotations_path in train_gold_annotations_paths:
        for fn in os.listdir(train_gold_annotations_path):
            tree = ET.parse(train_gold_annotations_path+'/'+fn)
            root = tree.getroot()
            (TEXT,TAGS) = [child for child in root]
            notetext = TEXT.text
            for child in TAGS:
    #             print(child.tag, child.attrib)
                row = child.attrib
                row['filename'] = fn
                row['text'] = notetext
                row['tag'] = child.tag
                rows.append(row)
    return pd.DataFrame(rows)
        
def get_dina_annotations():
    annotations = pd.DataFrame()

    bad=0
    for dina_annotations_path in ['../../n2c2/evens/' , '../../n2c2/odds/' ]:
        for dir in [dina_annotations_path+dir+'/' for dir in os.listdir(dina_annotations_path)]:
#             print(dir)
            RF = dir.split('/')[-1]
            if os.path.isdir(dir):
#                 print(dir)
                fns = [dir+ x for x in os.listdir(dir)]# +[dir+'/even/'+ x for x in os.listdir(dir+'/even/')]
#                 random.shuffle(fns)
                for fn in fns:
                    if '.txt' in fn:
                        with open(fn, 'r') as notefile:
                            notetext = notefile.read()
                        fn_base = fn.split('.txt')[0]
                        ann_fn = fn_base +'.ann'
                        ann_df =  pd.read_csv(ann_fn, sep='\t', names=['TA','TYPE','TEXT'])
                        entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann_fn)
                        if not relations=={}:
                            return relations
                        attr_df =  pd.DataFrame([([getattr(v, x) for x in vars(v)]) for _,v in attributes.items() ], columns=['id', 'type','target', 'values'])
                        entities_df =  pd.DataFrame([([getattr(v, x) for x in vars(v)]) for _,v in entities.items() ], columns=['id', 'type','span', 'text'])
                        entities_df['note_text']= [notetext]*len(entities_df)
                        entities_df['filename']=fn.split('/')[-1]
                        entities_df['RF'] = dir.split('/')[-2]
                        if not attr_df.empty:
                            entities_df['values'] = entities_df['id'].map(attr_df.groupby('target').apply(lambda x: list(x['values'])))
                            entities_df['types'] = entities_df['id'].map(attr_df.groupby('target').apply(lambda x: list(x['type'])))
                        annotations = pd.concat([annotations,entities_df])
                        
    return annotations
                        #                     return attr_df
    #                     return entities_df

dina_annotations = get_dina_annotations()

annotations_df, gold_df = get_complete_annotations()

smoker_override = {"221-04":"current","241-01": "unknown","242-02":"past","246-02":"current", "246-03": "unknown","277-03":"never","279-01":"ever",
                            "279-02":"current","291-05":"past", "295-01": "ever","296-03":"past","298-05":"unknown","303-04":"ever",
                                                    "320-01": "unknown","320-02":"unknown","320-03": "unknown","321-03":"never","322-01": "never","399-01": "current","399-02":"never"}
tagdict = {'CADMention':'CAD', 'HyperlipidemiaMention':'HYPERLIPIDEMIA', 'SmokerMention':'SMOKER', 'CADEvent':'CAD',
       'ObeseMention':'OBESE', 'DiabetesMention':'DIABETES', 'LDL':'HYPERLIPIDEMIA', 'CADTestResult':'CAD',
       'Cholesterol':'HYPERLIPIDEMIA', 'BMI':'OBESE', 'Medication2':'MEDICATION', 'Glucose':'DIABETES', 'A1C':'DIABETES',
       'BloodPressure':'HYPERTENSION', 'CADSymptom':'CAD', 'HypertensionMention':'HYPERTENSION', 'Medication1':'MEDICATION'}

def transform_brat_df_row(row):
    
    if type(row['types'])==float:
        row['types']=[]
    if 'Invalid' in row['types'] or 'Negation' in row['types']:
        return np.nan
    rowdict = {x:np.nan for x in ['id', 'time', 'type1', 'type2', 'filename', 'note_text', 'tag','observation_number', 'evidence_spans', 'text_cues', 'status', 'indicator', 'start', 'end', 'text', 'TYPE']}
    rowdict['tag'] = tagdict[row['RF']]
    rowdict['filename'] = row['filename'].replace('txt','xml')
    rowdict['note_text']=row['note_text']
    if 'Time_During'in row['types']:
        rowdict['time']='during DCT'
    elif 'Time_After'in row['types']:
        rowdict['time'] = 'after DCT'
    elif 'Time_Before' in row['types']:
        rowdict['time']= 'before DCT'
    else:
        rowdict['time']=np.nan
    if 'Medication' in row['RF']:
        if row['type'].startswith('Gold_') :
            drugs=row['type'][5:].replace('_',' ').split('+')
        elif row['type'].startswith('Gold') :
            drugs=row['type'][4:].replace('_',' ').split('+')
        else:
            drugs =row['type'].replace('_',' ').split('+')
        rowdict['type1']=drugs[0]
        if len(drugs)==2:
            rowdict['type2'] =drugs[1]
        rowdict['tag']='MEDICATION'
    elif 'Smoker' in row['RF']:
        rowdict['tag']='SMOKER'
        rowdict['status'] = 'brat' 
        if row['filename'].split('.')[0] in smoker_override.keys():
            rowdict['status'] = smoker_override[row['filename'].split('.')[0]]
            print('smoker override'+row['filename'])
    rowdict['evidence_spans'] = list(row['span'])
    rowdict['text_cues'] = [row['text']]
    return rowdict
    
dina_rows = dina_annotations.apply(transform_brat_df_row,axis=1)
dina_df = pd.DataFrame(list(dina_rows.dropna()))
gold_df=pd.concat([gold_df, dina_df])



def get_label(row):
    if 'before' in str(row['time']):
        label='before'
    else:
        if row['tag']=='MEDICATION':
            label =  row['tag']#+row['type1']#+row['time'] 

        elif row['tag']== 'SMOKER':
            if row['status']=='never' or row['status']=='unknown':
                label='before'
            else:
                label = 'SMOKER'
        elif row['tag'] =='FAMILY_HIST':
            label =  row['tag']
        elif row['tag'] =='PHI':
            label =  'before'#row['tag']#+row['TYPE']
        elif row['tag']== 'HYPERTENSION':
            label= row['tag']#+row['time']
        elif row['tag']=='DIABETES':
            label= row['tag']
        elif row['tag']=='HYPERLIPIDEMIA':
            label= row['tag']
        elif row['tag']=='CAD':
            label= row['tag']
        elif row['tag']== 'OBESE':
            label= row['tag']

    return [label]*len(row['evidence_spans'])


gold_df['labels'] = gold_df.apply(lambda row:get_label(row), axis=1)

def unravel(lst):
    return [i for j in lst for i in j]

def get_encoded_indices(row):
    ret = []
    for span in row['evidence_spans']:
        ret.append( [row['encoded'].char_to_token(i) for i in range(span[0],span[1])] )
    return ret
def get_all_labels(row):
    ret = ['O']*len(row['encoded']['input_ids'])
    for i,span in enumerate(row['evidence_spans']):
        ids =  [row['encoded'].char_to_token(j) for j in range(span[0],span[1])] 

        for id in ids:
            if id is not None:
                if row['labels'][i] is not 'before':
                    ret[id]='I-'+row['labels'][i]
#         for id in ids:
#             if id is not None:
#                 ret[id]='B-'+row['labels'][i]
#                 break
    if ret[0]!='O':
        ret[0] = 'B'+ret[0][1:]
    if ret[-1]!='O' and ret[-2]!=ret[-1]:
        ret[-1] = 'B'+ret[-1][1:]
    for i in range(1,len(ret)-1):
        if ret[i-1][1:]!=ret[i][1:] and ret[i]!='O':
            ret[i] = 'B'+ret[i][1:]
    return ret

grouped =gold_df.groupby('note_text')



# grouped['note_text'].apply(lambda x: x.iloc[0])
notes_df= grouped.agg({'evidence_spans':unravel, 'labels':unravel,'filename':lambda x: x.iloc[0] }).reset_index()
# notes_df[notes_df['note_text'].apply(lambda x: 'LAUREN' in x)]

# notes_df.columns=['note_text', 'labels','evidence_spans']
notes_df['encoded'] = notes_df['note_text'].apply(lambda x: tokenizer.encode_plus(x, truncation=False, is_split_into_words=False, padding='max_length', max_length=512) )

notes_df['encoded'].apply(lambda x: len(x['input_ids'])).mean()

#  gold_df.groupby('note_text')['note_text'].apply(lambda x:x).iloc[0]
notes_df

notes_df['token_labels'] = notes_df.apply(lambda row: get_all_labels(row), axis=1)

# [tokenizer.decode([x]) for x in notes_df.iloc[787]['encoded']['input_ids']]
# print(notes_df.iloc[787]['note_text'])
# list(zip([tokenizer.decode([x]) for x in notes_df.iloc[0]['encoded']['input_ids']],notes_df.iloc[0]['token_labels']))
# list(zip(notes_df.iloc[0]['evidence_spans'],notes_df.iloc[0]['labels']))
# notes_df.iloc[787]['note_text']
# notes_df.iloc[787]['encoded'].char_to_

labels_list =['B-CAD', 'B-DIABETES', 'B-FAMILY_HIST', 'B-HYPERLIPIDEMIA', 'B-HYPERTENSION', 'B-MEDICATION', 'B-OBESE', 'B-PHI', 'B-SMOKER', 'I-CAD', 'I-DIABETES', 'I-FAMILY_HIST', 'I-HYPERLIPIDEMIA', 'I-HYPERTENSION', 'I-MEDICATION', 'I-OBESE', 'I-PHI', 'I-SMOKER', 'O']

print(labels_list)

labels_dict = {k:i for i,k in enumerate(labels_list)}

from transformers import BertTokenizer, TFBertForTokenClassification
import tensorflow as tf
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',from_pt=True, num_labels=len(labels_list))
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
input_ids = inputs["input_ids"]
inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1
outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits
model.compile()

tokenizer(tokenizer.pad_token)

# [list(np.split(lst, range(510, lst.shape[0], 510)))
# notes_df

def chunk_ids(lst):
    lst=lst[1:-1]
    chunks=np.split(lst, range(510, len(lst), 510))
    ret = []
    for chunk in chunks:
        if len(chunk)<510:
            chunk=list(chunk)+[0]*(510-len(chunk))
        ret.append( np.array([101]+list(chunk)+[102]))
    return ret
def chunk_labels(lst):
    lst=lst[1:-1]
    chunks=np.split(lst, range(510, len(lst), 510))
    ret = []
    for chunk in chunks:
        if len(chunk)<510:
            chunk=list(chunk)+['O']*(510-len(chunk))
        ret.append( np.array(['O']+list(chunk)+['O']))
    return ret

def chunk_ids_with_overlap(lst):
    if len(lst)<=512:
        return chunk_ids(lst)
    return chunk_ids(lst)+chunk_ids(lst[256:])
def chunk_labels_with_overlap(lst):
    if len(lst)<=512:
        return chunk_labels(lst)
    return chunk_labels(lst)+chunk_labels(lst[256:])


# chunk_tokens(notes_df['token_labels'].iloc[0])
# chunk_ids(lst)

notes_df['chunked_tokens'] = notes_df['encoded'].apply(lambda x: chunk_ids_with_overlap(x['input_ids']))

notes_df['chunked_labels'] = notes_df['token_labels'].apply(lambda x: chunk_labels_with_overlap(x))

# notes_df['chunked_labels'].iloc[i]

# i=5
# j=-1
# print(len(notes_df['token_labels'].iloc[i]))
# pd.DataFrame(zip([tokenizer.decode([x]) for x in notes_df['chunked_tokens'].iloc[i][j]],notes_df['chunked_labels'].iloc[i][j])).tail(50)

# notes_df['encoded_tf']=notes_df['encoded'].apply(lambda x: tf.expand_dims( x.convert_to_tensors(tensor_type='tf')['input_ids'],axis=0 )    )

# notes_df['encoded'].iloc[2]

train_data = pd.DataFrame(list(zip(unravel(notes_df['chunked_tokens']), unravel(notes_df['chunked_labels']))), columns=['chunked_tokens','chunked_labels'])

train_data['label_ids'] = train_data['chunked_labels'].apply(lambda x: [labels_dict[y] for y in x])

train_data['filename']=unravel(unravel(notes_df.apply(lambda row: [[row['filename']]*len(row['chunked_labels'])],axis=1)))

train_inputs_dict = {'input_ids':np.array([x for x in train_data[train_data['filename'].apply( lambda x: x not in val_filenames)]['chunked_tokens']])}
val_inputs_dict = {'input_ids':np.array([x for x in train_data[train_data['filename'].apply( lambda x: x in val_filenames)]['chunked_tokens']])}

train_dataset = tf.data.Dataset.from_tensor_slices((
    train_inputs_dict,
    list(train_data[train_data['filename'].apply( lambda x: x not in val_filenames)]['label_ids'])
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_inputs_dict,
    list(train_data[train_data['filename'].apply( lambda x: x in val_filenames)]['label_ids'])
))

optimizer =  tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08, clipnorm=1)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer)

class MetricsCallback(Callback):
    def __init__(self):
         pass
    def on_epoch_end(self, epoch, logs):
#         try:
        self.model.save_pretrained('NER/models/clinicalbert_batchsize8_complete_brat_test'+str(epoch)+'.h5')
#         except Exception as e:
#             print(concept, e)
#             warnings.warn(concept+' '+str(e))
metrics_callback = MetricsCallback()


batch_size=8
model.fit(train_dataset.shuffle(10000).batch(batch_size), epochs=100,validation_data=val_dataset.batch(batch_size),callbacks=[metrics_callback])
