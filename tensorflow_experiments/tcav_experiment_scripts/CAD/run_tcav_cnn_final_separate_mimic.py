import time
import warnings
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils as utils    
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
import os 
import tensorflow as tf
import shutil
import argparse

layer_number='dense'
ckpt='epoch_0.ckpt'

print('REMEMBER TO UPDATE YOUR_PATH (where images, models are)!')
print('', flush=True)
time.sleep(.2) 
checkpoint_name = ckpt
model_to_run = checkpoint_name
user = 'projects'
project_name = 'tcav_class_text_bert'
working_dir = '/lscratch/'+os.environ['SLURM_JOB_ID']+'/' + project_name
activation_dir =  working_dir+ '/activations/'
cav_dir = 'cavs_CAD_relative/'
concepts_dir = './concept_listfiles'

source_dir = ''
scores_dir='cnn_scores/'

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)
utils.make_dir_if_not_exists(scores_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

target = 'CAD_final'  
concepts = []
concepts = concepts +  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
concepts = concepts + [x for x in os.listdir(concepts_dir) if x.startswith('mimic')]
scoresstr = []#''.join([fn for fn in os.listdir('cnn_scores') if 'distilbert_final_n2c2' in fn])
concepts = [concept for concept in concepts if not concept in scoresstr and not 'encode' in concept and not 'CAD' in concept and 'exclusive' not in concept]
# Create TensorFlow session.
print('concepts', concepts)
sess = utils.create_session()

GRAPH_PATH = '../tcav/cnn__all_epochs/'+checkpoint_name


LABEL_PATH = "resources/labels_map_final.txt"
print('before loading model', flush=True)
time.sleep(.2) 
mymodel = model.KerasModelWrapper(sess,GRAPH_PATH,LABEL_PATH)
print('after loading model', flush=True)
time.sleep(.2) 

print('bottlenecks',mymodel.bottlenecks_tensors)

## Step 3: Implement a class that returns activations (maybe with caching!)
print('before act gen', flush=True)
time.sleep(.2) 
act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=2000, document_encoder = act_gen.cnn_encode)
print('afteract gen', flush=True)
time.sleep(.2) 
## Step 4: Run TCAV and visualize concept importance

tf.compat.v1.logging.set_verbosity(0)
num_random_exp=50

print('before tcav', flush=True)
time.sleep(.2) 
all_bottlenecks = mymodel.bottlenecks_tensors
bottlenecks=[str(layer_number)]

mytcav = tcav.TCAV(sess,
                  target,
                  concepts,
                  bottlenecks,
                  act_generator,
                  alphas,
                  cav_dir=cav_dir,
                  num_random_exp=num_random_exp)
                  # random_concepts = concepts)
print ('This may take a while... Go get coffee!')
results = mytcav.run(run_parallel=False)
print ('done!')
print('after tcav', flush=True)
time.sleep(.2) 

import json
with open(scores_dir+'weekday_nocad_results_cnn_final_mimic_separate_rand10_savecavs'+str(layer_number) + ckpt.split('/')[-1] + '.json', 'w') as jfile:
   jfile.write(json.dumps(results))
utils_plot.plot_results(results, random_concepts=concepts, name='weekday_nocad_cnn_final_mimic_separate_rand10'+str(layer_number)+ckpt.split('/')[-1])
