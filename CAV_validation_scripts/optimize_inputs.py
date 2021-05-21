from transformers import DistilBertConfig,  DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf

import pickle
import os
num_random=50
vecs = []
rf = 'OASIS_discharge_meanbp_score'
cav_dir = 'cavs_distilbert_relative/'
for fn in os.listdir(cav_dir):
    if rf in fn:
#         print(fn)
        with open(cav_dir + fn ,'rb') as f:
            concept_vec=pickle.load(f)
        if concept_vec['concepts'][0]==rf:
            vecs.append(concept_vec['cavs'][0])
        elif concept_vec['concepts'][1]==rf:
            vecs.append(concept_vec['cavs'][1])
        else:
            assert False, concept_vec
concept_vec = np.mean(np.array(vecs),axis=0)

tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')

ckpt_dir='checkpoints/distilbert_scdropout0.5attdropout0.5drouout0.5_weighted/'
ckpt='distilbert_mortality_weighted3.h5'
print('loading model', flush=True)
model = TFDistilBertForSequenceClassification.from_pretrained(ckpt_dir+ckpt, output_hidden_states=True)

#https://blog.fastforwardlabs.com/2020/06/22/how-to-explain-huggingface-bert-for-question-answering-nlp-models-with-tf-2.0.html

embedding_matrix = model.distilbert.embeddings.word_embeddings.weight

# embedding_matrix.weight



# encoded_tokens =  tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="tf")
vocab_size = embedding_matrix.get_shape()[0]
token_ids = np.random.randint(vocab_size, size=512)

# convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
token_ids_tensor = tf.constant([token_ids], dtype='int32')
token_ids_tensor_one_hot = tf.Variable(tf.one_hot(token_ids_tensor, vocab_size))

token_ids_tensor_one_hot

# tf.nn.softmax(optimal_inputs,axis=2).numpy()

optimal_loss=999

#https://stackoverflow.com/questions/55552715/tensorflow-2-0-minimize-a-simple-function
opt = tf.optimizers.Adam(learning_rate=0.001)
optimal_inputs = token_ids_tensor_one_hot.numpy()
# optimal_loss = loss.numpy().mean()
print('starting loop',flush=True)
for i in range(100000):
    with tf.GradientTape(persistent=False) as tape:
    # (i) watch input variable
        tape.watch(token_ids_tensor_one_hot)

    
    
        # multiply input model embedding matrix; allows us do backprop wrt one hot input 
        inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix)  

        # (ii) get prediction
        preds = model({"inputs_embeds": inputs_embeds })
    #     answer_start, answer_end = get_best_start_end_position(start_scores, end_scores)

        loss = tf.keras.losses.mse(concept_vec.reshape(512,768),preds['hidden_states'][-1])
        # (iii) get gradient of input with respect to both start and end output
        gradient_non_normalized = tape.gradient(loss, token_ids_tensor_one_hot)

        opt.apply_gradients(zip([gradient_non_normalized], [token_ids_tensor_one_hot]))
        avg_loss = loss.numpy().mean()
        print(avg_loss,flush=True)
        if avg_loss<optimal_loss:
            optimal_loss=avg_loss
            optimal_inputs=token_ids_tensor_one_hot.numpy()

#         print(token_ids_tensor_one_hot.numpy()[0][0][0])
    #     print(preds['hidden_states'][-1][0][0][0])


import json
with open('input_optimization/first_results.json','w') as fp:
    json.dump(optimal_inputs.tolist(),fp)