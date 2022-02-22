import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
import os
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score, classification_report,average_precision_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, recall_score, accuracy_score
from scipy.special import expit as sigmoid

dirs_dct = dict(list(pd.read_csv('../directory_paths.csv')['paths'].apply(eval)))

checkpoints_dir=dirs_dct['checkpoints_dir']



bert_tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", model_max_length=512)

EXP_NAME = 'FINALLogisticCCS_halfsplit'

# wandb.init(settings=wandb.Settings(start_method="fork"))

for i in range(100):
    if EXP_NAME +'_'+str(i)+'.csv' not in os.listdir(checkpoints_dir):
        EXP_NAME = EXP_NAME +'_'+str(i)
        break
print('experiment name is '+ EXP_NAME)


save_dir = os.path.join(checkpoints_dir,EXP_NAME)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def bert_encode(data):
    encoded =  bert_tokenizer.encode_plus(
            text=data,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True,
            truncation=True
            )
    encoded=encoded['input_ids']
    encoded = encoded 
    encoded = encoded + [bert_tokenizer.pad_token_id]*(512-len(encoded))
    return np.asarray(encoded)

train_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_train_plus_structured.csv'))
val_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_val_plus_structured.csv'))
test_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_test_plus_structured.csv'))

train_df['encoded'] = train_df['TEXT'].apply(bert_encode)
val_df['encoded'] = val_df['TEXT'].apply(bert_encode)

test_df['encoded'] = test_df['TEXT'].apply(bert_encode)

train_df['TEXT']= train_df['encoded'].apply(lambda x: bert_tokenizer.decode(x))

val_df['TEXT']= val_df['encoded'].apply(lambda x: bert_tokenizer.decode(x))

test_df['TEXT']= test_df['encoded'].apply(lambda x: bert_tokenizer.decode(x))


def create_bag_of_words(encoded_text):
    bow = [0]*len(bert_tokenizer.vocab)
    for x in encoded_text:
        bow[x]=1
    return bow

len(bert_tokenizer.vocab.values())
# list(bert_tokenizer.vocab.values())[-1]

train_df['label']=train_df['label'].astype(int)
val_df['label']=val_df['label'].astype(int)
test_df['label']=test_df['label'].astype(int)


train_inputs = train_df['encoded'].apply(create_bag_of_words)
val_inputs = val_df['encoded'].apply(create_bag_of_words)
test_inputs = test_df['encoded'].apply(create_bag_of_words)

train_inputs = np.stack(train_inputs.values)
val_inputs = np.stack(val_inputs.values)
test_inputs = np.stack(test_inputs.values)

# def train_logistic_regression(features, label):
#     print ("Training the logistic regression model...")
#     from sklearn.linear_model import LogisticRegression
#     ml_model = LogisticRegression(C = 100,random_state = 0)
#     ml_model.fit(features, label)
#     print ('Finished')
#     return ml_model

# ml_model = train_logistic_regression(train_inputs, train_df['label'])

# print(pd.DataFrame(zip(vocab,ml_model.coef_[0] ), columns=['token', 'coeff']).sort_values(by='coeff', ascending=False).iloc[0:50].to_latex(index=False, float_format="%.2f"))
# df2 = pd.DataFrame(zip(vocab,ml_model.coef_[0]), columns=['token', 'coeff']).sort_values(by='coeff', ascending=False).iloc[10:20]

# df1.reset_index(drop=True).merge(df2.reset_index(drop=True), left_index=True, right_index=True)

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import nn, optim
from torch.nn import functional as F

# Defining neural network structure
class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # needs to be done everytime in the nn.module derived class
        super(BoWClassifier, self).__init__()

        # Define the parameters that are needed for linear model ( Ax + b)
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec): # Defines the computation performed at every call.
        # Pass the input through the linear layer,
        # then pass that through log_softmax.

        return F.log_softmax(self.linear(bow_vec), dim=1)

batch_size = 100
n_ui = 3000
epochs = 100
input_dim = len(bert_tokenizer.vocab)
output_dim = 2
lr_rate = 0.001

bow_nn_model = BoWClassifier(output_dim, input_dim)
# bow_nn_model.to(device)

# Loss Function
loss_function = nn.NLLLoss()
# Optimizer initlialization
optimizer = optim.SGD(bow_nn_model.parameters(), lr=0.01)

train_df['label'].dtype

train_dataset = list(zip(train_inputs, train_df['label']))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = list(zip(val_inputs, val_df['label']))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataset = list(zip(test_inputs, test_df['label']))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

import time
start_time = time.time()

# Train the model
best_mcc = 0
for epoch in range(100):
    for i, (ex, label) in enumerate(train_loader):
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        bow_nn_model.zero_grad()

        # Step 2. Make BOW vector for input features and target label
        bow_vec = ex.float()
        target = label
        
        # Step 3. Run the forward pass.
        probs = bow_nn_model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(probs, target)
        loss.backward()
        optimizer.step()
    if epoch%1==0:
        # calculate Accuracy
        metrics_dict={}
        for name, data_iter in {'val_':val_loader, 'test_':test_loader}.items():        
            correct = 0
            total = 0
            all_targets = []
            all_preds = []
            all_probs = []
            for i, (ex, label) in enumerate(data_iter):
                bow_vec = ex.float()
                target = label
                probs = bow_nn_model(bow_vec)

                _, predicted = torch.max(probs.data, 1)
                total+= target.size(0)
                all_targets = all_targets + list(target)
                all_preds = all_preds + list(predicted)
                all_probs = all_probs + probs.tolist()
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == target).sum()
            accuracy = 100 * correct/total
            
            metrics_dict[name+'accuracy'] = accuracy
            metrics_dict[name+'mcc'] = matthews_corrcoef(all_targets, all_preds)
            probs=sigmoid(all_probs)
            all_targets_2d = [[1,0] if x==0 else [0,1] for x in all_targets]
            metrics_dict[name+'accuracy'] =  accuracy_score(all_targets, all_preds)
            # print(all_targets)
            # print(all_preds)
            metrics_dict[name+'auc_avg'] = roc_auc_score(all_targets_2d, probs)
            metrics_dict[name+'auprc_avg'] = average_precision_score(all_targets_2d, probs)
            metrics_dict[name+'auc_1'] = roc_auc_score(all_targets, probs[:,1])
            metrics_dict[name+'auprc_1'] = average_precision_score(all_targets, probs[:,1])

            class_report = classification_report(all_targets,all_preds,output_dict=True) 
            metrics_dict[name+'precision'] = class_report['1']['precision']
            metrics_dict[name+'recall'] = class_report['1']['recall']
            metrics_dict[name+'f1-score'] = class_report['1']['f1-score']
            metrics_dict[name+'support'] = class_report['1']['support']
            metrics_dict[name+'specificity'] = class_report['0']['recall']
            metrics_dict['epoch'] = epoch
        fn = EXP_NAME+'_metrics.csv'
        dir = os.path.join(checkpoints_dir,EXP_NAME)
        df_dict = {k:[v] for k,v in metrics_dict.items()}
        if fn not in os.listdir(dir):
            pd.DataFrame(df_dict).to_csv(dir+fn, mode='a',header=True)
        else:
            pd.DataFrame(df_dict).to_csv(dir+fn, mode='a',header=False)
        mcc = metrics_dict['val_mcc']

        print("Iteration: {}. Loss: {}. Accuracy: {}. MCC: {}".format(epoch, loss.item(), accuracy, mcc))
        if mcc>best_mcc:
            best_mcc = mcc
            print('saving', epoch)
            save(bow_nn_model, os.path.join(checkpoints_dir,EXP_NAME), 'best', epoch)
            
print("Time taken to train the model: " + str(time.time() - start_time))
