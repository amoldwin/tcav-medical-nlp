#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/moldwinar/joboutputs/job_%j.out
#SBATCH --error=/data/moldwinar/joboutputs/job_%j.error
#SBATCH --time=5-9:00:00
#SBATCH --gres=gpu:v100x:1,lscratch:500

python tcav_experiment.py --ckpt_path /data/moldwinar/torch_tcav/model_train_scripts/checkpoints/FINALWeightedBlueBert512NicuLosAdmissions_2/checkpoint-26015 \
--num_trials 500 \
--bottleneck_name pool \
--experiment_name FINALWeightedBlueBert512NicuLosAdmissions_2_checkpoint-26015_500_pool \
--working_dir /data/moldwinar/torch_tcav/tcav_scripts/FINALWeightedBlueBert512NicuLosAdmissions_2_checkpoint-26015_500_pool \
--source_dir /data/moldwinar/tcav2/ \
--results_dir /data/moldwinar/torch_tcav/tcav_scripts/results/FINALWeightedBlueBert512NicuLosAdmissions_2_checkpoint-26015_500_pool \
--target nicu_los \
--activations_dir LSCRATCH \
--all_activations_dir LSCRATCH/all_activations/ \
--concepts_dir /data/moldwinar/tcav2/concept_listfiles/NICU \
--data_encode_function_name bert_encode \
--wrapper_name BertWrapper \
--concepts_list "['positive_Septicemia', 'positive_Jaundice', 'positive_Other perinatal conditions', 'positive_Suspected Infection', 'positive_Tuesday', 'positive_Premature birth', 'positive_Patent ductus arteriosus', 'positive_Needs Hepatitis Vaccination', 'positive_Chronic respiratory disease', 'positive_MALE', 'positive_FEMALE', 'positive_Single Birth, Cesarean Section', 'positive_Sunday', 'positive_Saturday', 'positive_Low birth weight', 'positive_Sleep Apnea', 'positive_Wednesday', 'positive_Anemia of prematurity', 'positive_Monday', 'positive_Retrolental fibroplasia', 'positive_3+ Birthmates, Cesarian Section', 'positive_Bradycardia', 'positive_Friday', 'positive_Twin Birth, Cesarian Section', 'positive_Respiratory Distress Syndrome', 'positive_Thursday']" 