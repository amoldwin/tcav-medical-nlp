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

print('before argparse')
parser = argparse.ArgumentParser()
parser.add_argument('--layer_number', type=int, default=5)
parser.add_argument('--ckpt_path', type=str, default='distilbert_cad_final1.h5' )
args = parser.parse_args()
print('after parse_args')
layer_number = args.layer_number
ckpt = args.ckpt_path
print('layer_number',layer_number)
num_random_exp=10

name='weekday_nocad_distilbert_final_mimic_separate__layer'+str(layer_number)+'num_random'+str(num_random_exp)+ckpt.split('/')[-1]


print('REMEMBER TO UPDATE YOUR_PATH (where images, models are)!')
print('', flush=True)
time.sleep(.2) 
checkpoint_name = ckpt
model_to_run = checkpoint_name
user = 'projects'
project_name = 'tcav_class_text_bert'
working_dir = '/lscratch/'+os.environ['SLURM_JOB_ID']+'/' + project_name
activation_dir =  working_dir+ '/activations/'
cav_dir = 'cavs_CAD_regular/'
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
concepts = [concept for concept in concepts if not concept in scoresstr and not 'encode' in concept and not 'CAD' in concept and not 'exclusive' in concept]
# Create TensorFlow session.
print('concepts', concepts)
sess = utils.create_session()

GRAPH_PATH = '../tcav/bertcheckpoints/distilbert/'+checkpoint_name


LABEL_PATH = "resources/labels_map_final.txt"
print('before loading model', flush=True)
time.sleep(.2) 
mymodel = model.DistilBertWrapper(sess,GRAPH_PATH,LABEL_PATH)
print('after loading model', flush=True)
time.sleep(.2) 

print('bottlenecks',mymodel.bottlenecks_tensors)

## Step 3: Implement a class that returns activations (maybe with caching!)
print('before act gen', flush=True)
time.sleep(.2) 
act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=1000, document_encoder = act_gen.distilbert_encode)
print('afteract gen', flush=True)
time.sleep(.2) 
## Step 4: Run TCAV and visualize concept importance

tf.compat.v1.logging.set_verbosity(0)

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
with open(scores_dir+ name +'_savecavs.json', 'w') as jfile:
   jfile.write(json.dumps(results))
utils_plot.plot_results(results, num_random_exp=num_random_exp, name=name)
