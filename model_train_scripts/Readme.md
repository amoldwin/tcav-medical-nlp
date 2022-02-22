Model Training Instructions:

Each model has its own training script. 

Most of the models use the HuggingFace data loader, but logistic regression and CNN models do not. 


The CNN scripts for the three CNN models (text cnn for NICU LOS, text CNN, for metabolic syndrome, and CNN with RMLP for metabolic syndrome) are in the "text_cnn" folder.

The uggingface models save the checkpoint with the best validation MCC as well as the final checkpoint. We use the one with the best validation MCC for our TCAV experiments.