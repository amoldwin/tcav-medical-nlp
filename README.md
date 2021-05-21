# tcav-medical-nlp
Concept-Level Interpretability for Models that Process Clinical Notes

Running TCAV model interpretability experiments on BERT NLP models trained on MIMIC-III ICU clinical notes, with concept categories identified using tabular data and text annotions.

This project is based on the TCAV interpretability framework developed by Kim et al. in (https://arxiv.org/pdf/1711.11279.pdf)
The original TCAV repository can be found here: https://github.com/tensorflow/tcav and is used mainly for computer vision/image classification models.
  
The tensorflow 2 fork that our code is based on is available at: https://github.com/monz/tcav (but I had to make some further edits for tensorflow 2 and Huggingface transformers to work properly.)

Our version of the benchmark that includes clinical note text was used in our work published at [...]
