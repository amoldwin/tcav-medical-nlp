# Model Training Instructions
We include scripts for training 7 architectures. With the exception of the CNN models, we used the Huggingface API to develop and train models and load datasets.
For the CNN models, we used code from: https://github.com/Shawn1993/cnn-text-classification-pytorch.

## Training non-CNN models
Each model has its own training script. 

(1) Train Logistic Regression on NICU Length of Stay:

(2) Train text MLP on NICU Lenth of Stay: 

(3) Train BERT on NICU Length of Stay:

(3) Train Logistic Regression on MetS:

(4) Train text MLP on NICU Lenth of MetS: 

(5) Train BERT on NICU Length of MetS:

(6) Train RMLP on MetS:

(7) Train Bert  + RMLP Siamese model on MetS:

##Training CNN Models

(1) Train Text CNN on NICU LoS

(2) Train Text CNN on MetS

(3) Train RMLP + CNN On MetS

Most of the models use the HuggingFace data loader, but logistic regression and CNN models do not. 


The CNN scripts for the three CNN models (text cnn for NICU LOS, text CNN, for metabolic syndrome, and CNN with RMLP for metabolic syndrome) are in the "text_cnn" folder.

The Huggingface models save the checkpoint with the best validation MCC as well as the final checkpoint. We use the one with the best validation MCC for our TCAV experiments.
