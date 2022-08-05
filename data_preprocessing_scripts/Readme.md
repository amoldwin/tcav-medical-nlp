# Dataset Creation
In this directory, you can find all of the scripts used to generate our task datasets for (1) NICU Length-of-Stay prediction and (2) adult Metabolic Syndrome prediction. In addition to generating the task datasets, these scripts will create csv files with examples for each of the clinical concepts that we test with TCAV.

Both tasks involve relying on the text of clinical notes to predict a binary outcome.

# Usage

A number of scripts will need to be run to generate the data. 
## NICU LoS:

(1) Run the first 2 cells of [```create_training_data_metabolic_syndrome.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/create_training_data_metabolic_syndrome.ipynb) (this is where we generated the merged admissions, so we need this for both tasks)

(1) run [```nicu_pull_birth_weight_gest_age.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/nicu_pull_birth_weight_gest_age.py) 

(2) run [```create_los_neonates_none_removed.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/create_los_neonates_none_removed.ipynb)



## Metabolic Syndrome
(1) As before, run the first 2 cells of [```create_training_data_metabolic_syndrome.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/create_training_data_metabolic_syndrome.ipynb)

(2) Run [```pull_chart_labs_data_metabolic_syndrome.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/pull_chart_labs_data_metabolic_syndrome.py) to output a file of the relevant lab and chart measurements

(3) Run the rest of the cells in the jupyter notebook [```aggregate_metabolic_structured_data.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/aggregate_metabolic_structured_data.ipynb) 

(4) Run the rest of [```create_training_data_metabolic_syndrome.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/data_preprocessing_scripts/create_training_data_metabolic_syndrome.ipynb) to generate training data and concept examples(pay attention to comments about optional steps)



In addition, we had some other minor preprocessing steps that included (1) generating a file listing all the section headers present in each clinical note in MIMIC and (2) Training an NER model (found in the "model train_scripts" folder) to supplement the examples for the "smoker status" concept. These steps ended up being time consuming and mostly inconsequential in affecting the final dataset cohorts, so for simplicity we will omit them in our instructions for reproducing the dataset.
