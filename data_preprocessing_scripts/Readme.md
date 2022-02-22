In this directory, you can find all of the scripts used to generate our task datasets for:

(1)Neonatal Intensive Care Unit Length of Stay and (2)Metabolic Syndrome detection.

Both tasks involve relying on the text of clinical notes to predict a binary outcome.

In addition to generating the task datasets, these scripts will create csv files with examples for each of the clinical concepts that we test with TCAV.

A number of scripts will need to be run to generate the data. 
For the NICU task:

(1)nicu_pull_birth_weight_gest_age.py

(2)create_los_neonates_none_removed.ipynb



For the metabolic syndrome task:
(1) Run the first 2 cells of create_training_data_metabolic_syndrome.ipynb

(2) Run pull_chart_labs_data_metabolic_syndrome.py to output a file of the relevant lab and chart measurements

(3) Run the rest of the cells in the jupyter notebook aggregate_metabolic_structured_data.ipynb 

(4) Run the rest of create_training_data_metabolic_syndrome.ipynb to generate training data and concept examples(pay attention to comments about optional steps)



In addition, we had some other minor preprocessing steps that included (1) generating a file listing all the section headers present in each clinical note in MIMIC and (2) Training an NER model (found in the "model train_scripts" folder) to supplement the examples for the "smoker status" concept. These steps ended up being time consuming and mostly inconsequential in affecting the final dataset cohorts, so for simplicity we will omit them in our instructions for reproducing the dataset.
