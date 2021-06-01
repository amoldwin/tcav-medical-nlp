import sys
sys.path.insert(0,'../tcav/')
import cav as cav
import model as model
import tcav as tcav
import utils as utils
import utils_plot as utils_plot # utils_plot requires matplotlib
import os
import torch
import activation_generator as act_gen
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import json

working_dir = './tcav_class_test'
activation_dir =  working_dir + '/activations/'
cav_dir = working_dir + '/cavs/'
source_dir = "../../tcav2/"
bottlenecks = ['11']
concepts_dir=source_dir+'/concept_listfiles' 
scores_dir = './results/'    
    
utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)
utils.make_dir_if_not_exists(scores_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

target = 'mort_30day'  

concepts = []
concepts = concepts +  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
concepts = ['WEEKDAY_discharge_'+x for x in concepts]
concepts = concepts + [x for x in os.listdir(concepts_dir) if x.startswith('filtered_disch')]

scoresstr = []#''.join([fn for fn in os.listdir('cnn_scores') if 'distilbert_final_n2c2' in fn])
concepts = [concept for concept in concepts if not concept in scoresstr and not 'encode' in concept]
# Create TensorFlow session.
print('concepts', concepts)# random_counterpart = 'random500_1'


LABEL_PATH = source_dir + "/resources/labels_map_30mort.txt"

ckpt='../model_train_scripts/checkpoints/WeightedBlueBert512DischargeMortality/checkpoint-12560'
mymodel = model.BertWrapper(model_path=ckpt, labels_path=LABEL_PATH)

act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, document_encoder=act_gen.bert_encode,max_examples=2000)

tf.logging.set_verbosity(0)

num_random_exp=10 # folders (random500_0, random500_1)

mytcav = tcav.TCAV(target,
                   concepts,
                   bottlenecks,
                   act_generator,
                   alphas,
                   cav_dir=cav_dir,
                   num_random_exp=num_random_exp)

results = mytcav.run()

with open(scores_dir+'scratch' + ckpt.split('/')[-1]+'_' +str(bottlenecks[0])+'_'+str(num_random_exp)+'randexp.json', 'w') as jfile:
   jfile.write(json.dumps(results))