#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/moldwinar/joboutputs/job_%j.out
#SBATCH --error=/data/moldwinar/joboutputs/job_%j.error
#SBATCH --time=5-9:00:00
#SBATCH --gres=gpu:v100x:1,lscratch:500

python tcav_experiment.py --ckpt_path /data/moldwinar/torch_tcav/model_train_scripts/checkpoints/FINALLogisticCCS_halfsplit_0/best_steps_13.pt \
--num_trials 500 \
--bottleneck_name pre_sum \
--experiment_name FINALLogisticCCS_halfsplit_0_best_steps_13.pt_500_pre_sum \
--working_dir /data/moldwinar/torch_tcav/tcav_scripts/FINALLogisticCCS_halfsplit_0_best_steps_13.pt_500_pre_sum \
--source_dir /data/moldwinar/tcav2/ \
--results_dir /results/FINALLogisticCCS_halfsplit_0_best_steps_13.pt_500_pre_sum \
--target CCS \
--activations_dir LSCRATCH \
--all_activations_dir LSCRATCH/all_activations/ \
--concepts_dir /data/moldwinar/tcav2/concept_listfiles/metabolic \
--data_encode_function_name logistic_bert_encode \
--wrapper_name LogisticWrapper \
--concepts_list "['positive_MED_ACE', 'positive_MED_INS', 'positive_OLD', 'positive_SMOKER', 'positive_MED_STAT', 'positive_MED_BETA', 'positive_MED_FIBR', 'positive_MALE', 'positive_FEMALE', 'positive_OBESE', 'positive_HYPERTENSION', 'positive_LOWHDL', 'positive_Monday', 'positive_Tuesday', 'positive_Wednesday', 'positive_DIABETES', 'positive_HYPERGLYCERIDEMIA', 'positive_YOUNG', 'positive_HYPERLIPIDEMIA', 'positive_Sunday', 'positive_Monday', 'positive_Thursday', 'positive_Friday', 'positive_Saturday']" 