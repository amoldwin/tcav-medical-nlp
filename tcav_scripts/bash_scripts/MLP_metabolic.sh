#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/moldwinar/joboutputs/job_%j.out
#SBATCH --error=/data/moldwinar/joboutputs/job_%j.error
#SBATCH --time=5-9:00:00
#SBATCH --gres=gpu:v100x:1,lscratch:500

python tcav_experiment.py --ckpt_path /data/moldwinar/torch_tcav/model_train_scripts/checkpoints/CCS_MLP_TEXT__0/checkpoint-14877 \
--num_trials 500 \
--bottleneck_name pool \
--experiment_name CCS_MLP_TEXT__0_checkpoint-14877_500_pool \
--working_dir /data/moldwinar/torch_tcav/tcav_scripts/CCS_MLP_TEXT__0_checkpoint-14877_500_pool \
--source_dir /data/moldwinar/tcav2/ \
--results_dir /results/CCS_MLP_TEXT__0_checkpoint-14877_500_pool \
--target CCS \
--activations_dir LSCRATCH \
--all_activations_dir LSCRATCH/all_activations/ \
--concepts_dir /data/moldwinar/tcav2/concept_listfiles/metabolic \
--data_encode_function_name bert_encode \
--wrapper_name MLPWrapper \
--concepts_list "['positive_MED_ACE', 'positive_MED_INS', 'positive_OLD', 'positive_SMOKER', 'positive_MED_STAT', 'positive_MED_BETA', 'positive_MED_FIBR', 'positive_MALE', 'positive_FEMALE', 'positive_OBESE', 'positive_HYPERTENSION', 'positive_LOWHDL', 'positive_Monday', 'positive_Tuesday', 'positive_Wednesday', 'positive_DIABETES', 'positive_HYPERGLYCERIDEMIA', 'positive_YOUNG', 'positive_HYPERLIPIDEMIA', 'positive_Sunday', 'positive_Monday', 'positive_Thursday', 'positive_Friday', 'positive_Saturday']" 