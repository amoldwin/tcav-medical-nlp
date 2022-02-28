#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH --output=/data/moldwinar/joboutputs/job_%j.out
#SBATCH --error=/data/moldwinar/joboutputs/job_%j.error
#SBATCH --time=5-9:00:00
#SBATCH --gres=gpu:v100x:1,lscratch:500

python tcav_experiment.py --ckpt_path /data/moldwinar/torch_tcav/model_train_scripts/checkpoints/logistic/best_steps_84.pt \
--num_trials 500 \
--bottleneck_name pre_sum \
--experiment_name logistic_best_steps_84.pt_500_pre_sum \
--working_dir /data/moldwinar/torch_tcav/tcav_scripts/logistic_best_steps_84.pt_500_pre_sum \
--source_dir /data/moldwinar/tcav2/ \
--results_dir /data/moldwinar/torch_tcav/tcav_scripts/results/logistic_best_steps_84.pt_500_pre_sum \
--target nicu_los \
--activations_dir LSCRATCH \
--all_activations_dir LSCRATCH/all_activations/ \
--concepts_dir /data/moldwinar/tcav2/concept_listfiles/NICU \
--data_encode_function_name logistic_bert_encode \
--wrapper_name LogisticWrapper \
--concepts_list "['positive_Septicemia', 'positive_Jaundice', 'positive_Other perinatal conditions', 'positive_Suspected Infection', 'positive_Tuesday', 'positive_Premature birth', 'positive_Patent ductus arteriosus', 'positive_Needs Hepatitis Vaccination', 'positive_Chronic respiratory disease', 'positive_MALE', 'positive_FEMALE', 'positive_Single Birth, Cesarean Section', 'positive_Sunday', 'positive_Saturday', 'positive_Low birth weight', 'positive_Sleep Apnea', 'positive_Wednesday', 'positive_Anemia of prematurity', 'positive_Monday', 'positive_Retrolental fibroplasia', 'positive_3+ Birthmates, Cesarian Section', 'positive_Bradycardia', 'positive_Friday', 'positive_Twin Birth, Cesarian Section', 'positive_Respiratory Distress Syndrome', 'positive_Thursday']" 