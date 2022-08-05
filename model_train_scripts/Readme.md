# Model Training Instructions
We include scripts for training 7 architectures. With the exception of the CNN models, we used the Huggingface API to develop and train models and load datasets.
For the CNN models, we used code from: https://github.com/Shawn1993/cnn-text-classification-pytorch.

## Training non-CNN models
Each model has its own training script. 

(1) Train Logistic Regression on NICU Length of Stay:
[```train_NICU_logistic.ipynb```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_NICU_logistic.ipynb)

(2) Train text MLP on NICU Lenth of Stay: 
[```train_NICU_MLP.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_NICU_MLP.py)

(3) Train BERT on NICU Length of Stay:
[```train_NICU_BERT.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_NICU_BERT.py)

(4) Train Logistic Regression on MetS:
[```train_CCS_logistic.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_CCS_logistic.py)

(5) Train text MLP on MetS: 
[```train_CCS_MLP.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_CCS_MLP.py)

(6) Train BERT on MetS:
[```train_CCS_bert.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_CCS_bert.py)

(7) Train RMLP on MetS:
[```train_CCS_just_res.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_CCS_just_res.py)

(8) Train Bert  + RMLP Siamese model on MetS:
[```train_CCS_RHN.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/train_CCS_RHN.py)

## Training CNN Models
The CNN scripts for the three CNN models (text cnn for NICU LOS, text CNN, for metabolic syndrome, and CNN with RMLP for metabolic syndrome) are in the "text_cnn" folder.

(9) Train Text CNN on NICU LoS
[```main_nicu_los.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/text_cnn/main_nicu_los.py)

(10) Train Text CNN on MetS
[```main_metabolic.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/text_cnn/main_metabolic.py)

(11) Train RMLP + CNN On MetS
[```main_metabolic_plus_rmlp.py```](https://github.com/amoldwin/tcav-medical-nlp/blob/main/model_train_scripts/text_cnn/main_metabolic_plus_rmlp.py)

Most of the models use the HuggingFace data loader, but logistic regression and CNN models do not. 




The Huggingface models save the checkpoint with the best validation MCC as well as the final checkpoint. We use the one with the best validation MCC for our TCAV experiments.
