import sys
sys.path.insert(0,'../tcav/')
import cav as cav
import model as model
import tcav as tcav
import utils as utils
import os
import torch
import activation_generator as act_gen
import tensorflow.compat.v1 as tf
import encode_functions
import pickle
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Source directories and settings for TCAV.')

parser.add_argument("--bottleneck_name", type=str,required=True)
parser.add_argument("--ckpt_path", type=str,required=True)
parser.add_argument("--working_dir", type=str,required=True)
parser.add_argument("--activations_dir", type=str,required=True)
parser.add_argument("--source_dir", type=str,required=True)
parser.add_argument("--target", type=str,required=True)
parser.add_argument("--concepts_dir", type=str,required=True)
parser.add_argument("--results_dir", type=str,required=True)
parser.add_argument("--all_activations_dir", type=str,required=True)
parser.add_argument("--wrapper_name", type=str,required=True)
parser.add_argument("--concepts_list", type=str,required=True)
parser.add_argument("--experiment_name", type=str,required=True)
parser.add_argument("--data_encode_function_name", type=str,required=True)
parser.add_argument("--num_trials", type=int,required=True)

args = parser.parse_args()
print('args',args)
args.concepts_list=eval(args.concepts_list)
if args.activations_dir=='LSCRATCH':
    args.activations_dir='/lscratch/'+os.environ['SLURM_JOB_ID']+'/'
    args.all_activations_dir=os.path.join(args.activations_dir,'all_activations')

concepts = args.concepts_list

all_negative_concepts = ['allnegative_'+x[9:] for x in concepts]

bottlenecks =[args.bottleneck_name]


ckpt = args.ckpt_path
print(ckpt)
working_dir = args.working_dir
lscratch_dir = args.activations_dir
activation_dir =  lscratch_dir + '/activations/'
cav_dir = os.path.join(working_dir,'cavs')
source_dir = args.source_dir
concepts_dir=args.concepts_dir
scores_dir =args.results_dir
encoded_dir = os.path.join(concepts_dir,'encoded_concepts',args.data_encode_function_name)
    
utils.make_dir_if_not_exists(args.all_activations_dir)

utils.make_dir_if_not_exists(activation_dir)
utils.make_dir_if_not_exists(working_dir)
utils.make_dir_if_not_exists(cav_dir)
utils.make_dir_if_not_exists(scores_dir)
utils.make_dir_if_not_exists(encoded_dir)

# this is a regularizer penalty parameter for linear classifier to get CAVs. 
alphas = [0.1]   

target = args.target


print('concepts', concepts)
wrappers={'BertWrapper':model.BertWrapper,
          'MLPWrapper':model.MLPWrapper,
          'CnnWrapper':model.CnnWrapper,
          'CnnPlusStructuredWrapper':model.CnnPlusStructuredWrapper,
          'LogisticWrapper':model.LogisticWrapper,
          'BertRHNWrapper':model.BertRHNWrapper,
          'JustRHNWrapper':model.JustRHNWrapper
         }
mymodel = wrappers[args.wrapper_name](model_path=ckpt, target=args.target)


#first generate activations for all required documents and save them
act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, args.activations_dir,
                                                 document_encoder=encode_functions.encode_fns_dct[args.data_encode_function_name],
                                                 max_examples=9999999, concept_dir=concepts_dir)



all_required_exs = pd.DataFrame()
for fn in concepts+all_negative_concepts:
   all_required_exs = pd.concat([all_required_exs,   pd.read_csv(os.path.join(concepts_dir,fn))], ignore_index=True).drop_duplicates(subset=['ROW_ID'])

all_required_exs.to_csv(os.path.join(args.concepts_dir,'all_required_exs.csv'))

if not os.path.isfile(os.path.join(args.all_activations_dir,'all_activations.pk')):

    all_activations = act_generator.get_activations_for_concept('all_required_exs.csv', bottleneck=bottlenecks[0],shuffled=False)

    all_activations_df = pd.DataFrame((zip(all_required_exs['ROW_ID'], all_activations)),columns=['ROW_ID','activations'])

    all_activations_df.to_pickle(os.path.join(args.all_activations_dir,'all_activations.pk'))

    
#now create a new activation generator that will just load pre-saved activations for each concept 
act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, None, from_row_id=True, document_encoder=act_gen.bert_encode,max_examples=1000, concept_dir=concepts_dir, all_acts_dir=args.all_activations_dir)

tf.logging.set_verbosity(0)

num_random_exp=args.num_trials


for i, concept in enumerate(concepts):
    negative_concepts = [all_negative_concepts[i]+'_'+str(j) for j in range(args.num_trials)]

    print('concept was '+concept+', negative concepts were: ', negative_concepts,flush=True)
    results_filename = '_'.join([args.experiment_name,concept,'.pk'])
    if os.path.isfile(os.path.join(args.results_dir,results_filename)):
        print('skipping concept because results already present')
        continue
    mytcav = tcav.TCAV(target,
                       [concept],
                       bottlenecks,
                       act_generator,
                       alphas,
                       cav_dir=cav_dir,
                       random_concepts=negative_concepts
                      )

    results = mytcav.run()

    with open(os.path.join(args.results_dir,results_filename), 'wb') as pkfile:
       pickle.dump(results, pkfile)
    print('results stored at', os.path.join(args.results_dir,results_filename) )
    
    

args.num_trials

