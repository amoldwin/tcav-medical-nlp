import tensorflow as tf
import pandas as pd
from transformers import BertTokenizerFast, BatchEncoding, TFBertForTokenClassification
import numpy as np
from sklearn.metrics import recall_score, classification_report


tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# val_df = pd.read_csv('datasets/n2c2_val_dataset.csv')
notes_df = pd.read_csv('../../../LHC_mimic/mimic3_1.4/raw_data/NOTEEVENTS.csv')
notes_df = notes_df[notes_df['CATEGORY'].apply(lambda x: x in ['Nursing/other','Nursing','General','Physician '])].sort_values(by='CHARTTIME')
# test_df = pd.read_csv('../data/CAD_test.csv')
model = TFBertForTokenClassification.from_pretrained('NER/models/clinicalbert_batchsize8_complete_brat_test2.h5')

# subset_all = notes_df.sample(230000)
s=10000
split_notes = [notes_df.iloc[s*i:s*i+s] for i in range(int(len(notes_df)))]
# subset = subset[subset['TEXT'].apply(lambda x: 'hyperten' in x[:510])]
for i, subset in enumerate(split_notes):
    # subset
    # if i<33:
    #     continue
    subset['encoded'] = subset['TEXT'].apply(lambda x: tokenizer.encode_plus(x, truncation=True, is_split_into_words=False, padding='max_length', max_length=512)  )

    # subset

    val_inputs_dict = {'input_ids':np.array([x['input_ids'] for x in subset['encoded']])}

    val_inputs_dict

    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_inputs_dict
    ))

    outputs=model.predict(val_dataset.batch(8))

    predictions = tf.argmax(outputs[0], axis=2)

    prediction_probabilities = tf.reduce_max(outputs[0], axis=2)

    labels_list =['B-CAD', 'B-DIABETES', 'B-FAMILY_HIST', 'B-HYPERLIPIDEMIA', 'B-HYPERTENSION', 'B-MEDICATION', 'B-OBESE', 'B-PHI', 'B-SMOKER', 'I-CAD', 'I-DIABETES', 'I-FAMILY_HIST', 'I-HYPERLIPIDEMIA', 'I-HYPERTENSION', 'I-MEDICATION', 'I-OBESE', 'I-PHI', 'I-SMOKER', 'O']

    reverse_labels_dict = {i:k for i,k in enumerate(labels_list)}

    val_results_df = pd.DataFrame([list(predictions.numpy())]).transpose()

    val_results_df['pred_probs'] = list(prediction_probabilities.numpy())

    val_results_df['tokens'] = list(subset['encoded'].apply(lambda x: [tokenizer.decode([y]) for y in x['input_ids']]) )

    val_results_df.columns=['class_nums','pred_probs','tokens']

    val_results_df['classes']=val_results_df['class_nums'].apply(lambda x: [reverse_labels_dict[y] for y in x] )

    def unravel(lst):
        return [i for j in lst for i in j]

    # i =0
    # result = pd.DataFrame(list(zip(val_results_df.iloc[i]['tokens'], val_results_df.iloc[i]['classes'],val_results_df.iloc[i]['pred_probs'])), columns=['tokens', 'token_preds', 'pred_score']).head(50)
    # result[result['token_preds'].apply(lambda x: x!='O')]

    val_results_df['ROW_ID']= list(subset['ROW_ID'])
    val_results_df['TEXT']= list(subset['TEXT'])
    # val_results_df['label']= subset['label']


    val_results_df.to_csv('results/cat_mimic_ner_results'+str(i)+'.csv')

