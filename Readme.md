# tcav-medical-nlp
Concept-Level Interpretability for Models that Process Clinical Notes

Running TCAV model interpretability experiments on BERT NLP models trained on MIMIC-III ICU clinical notes, with concept categories identified using tabular data and text annotions.

This project is based on the TCAV interpretability framework developed by Kim et al. in (https://arxiv.org/pdf/1711.11279.pdf)
The original TCAV repository can be found here: https://github.com/tensorflow/tcav and is used mainly for computer vision/image classification models.
  
The pytorch fork that our code is based on is available at: https://github.com/rakhimovv/tcav 

Our work using TCAV to interpret NLP-based clinical prediction models is published at [...]


Instructions:

(0) Clone the repository and install the required python libraries using the following 2 commands:
 ```bash
 $ pip -r requirements.txt
 ```

(1) Run through the jupyter notebook ```specify_directories.ipynb```, making sure to change the paths to your desired locations for all of the subsequent data, model checkpoints, and results to be stored. 

This way you can hopefully specify the paths just once. This is supposed to be instead of having hardcoded filepaths in the experiment scripts, but there may be a few paths left in the experiment scripts that I missed.


(2) Generate data for tasks and concepts


Follow instructions in the readme in the "data_preprocessing_scripts" directory.


(3) Train models


Follow instructions in the readme in the "model_train_scripts" directory.


(4) Run TCAV interpretability to interpret clinical NLP models in terms of concepts as they are expressed in clinical notes. Then Visualize results.


Follow instructions in the readme in the "tcav_scripts" directory.


(5) Run word-level interpretability (integrated gradients and average per-word attention) to compare the two paradigms.


Follow instructions in the readme in the "word_level_explanations" directory.
