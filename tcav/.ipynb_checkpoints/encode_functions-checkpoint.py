import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import numpy as np

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def cnn_encode(text_ex):
    with open('resources/tokenizer_final.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return pad_sequences(tokenizer.texts_to_sequences([text_ex]), padding='post', maxlen=4096)[0]

def bert_encode(data):
    encoded = bert_tokenizer.encode_plus(
        text=data,  # Preprocess sentence
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        return_attention_mask=True,  # Return attention mask
        return_token_type_ids=True,
        truncation=True
    )
    encoded = encoded['input_ids']
    encoded = encoded
    encoded = encoded + [bert_tokenizer.pad_token_id] * (512 - len(encoded))
    return np.asarray(encoded)


def bert_rhn_encode(data):
    encoded = bert_tokenizer.encode_plus(
        text=data,  # Preprocess sentence
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        return_attention_mask=True,  # Return attention mask
        return_token_type_ids=True,
        truncation=True
    )
    encoded = encoded['input_ids']
    encoded = encoded
    encoded = encoded + [bert_tokenizer.pad_token_id] * (512 - len(encoded))
    return np.asarray(encoded)


def create_bag_of_words(encoded_text):
    bow = [0] * len(bert_tokenizer.vocab)
    for x in encoded_text:
        bow[x] = 1
    return bow


def logistic_bert_encode(data):
    return create_bag_of_words(bert_encode(data))
encode_fns_dct = {x.__name__:x for x in [cnn_encode,bert_encode,bert_rhn_encode,logistic_bert_encode]}

