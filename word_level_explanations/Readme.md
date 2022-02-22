Word Level Explanations:

Instructions are still being written. This part of the code hasn't been refactored yet.

In the paper, we provide the following word-level explanations for the BERT NICU LOS model:

(1) Top 25 words with highest integrated gradients, when averaged over all occurrences of the word in the test set.

Instructions:
    
    (A) Run get_grads_NICU.py to calculate integrated gradients for the entire test set
    
    (B) Compile list of best words using usage_example_nicu.ipynb
    
(2) Top 25 words with highest transformer attention, when averaged over all occurrences of the word in the test set. 

    (A) Run get_nicu_attentions.py to calculate average attentions for the entire train set
    
    (B) Compile list of best words using  get_nicu_attentions.ipynb
    
(3) Visualizations of excertps from specific clinical notes, showing a heatmap of the words weighted by each word's integrated gradients.