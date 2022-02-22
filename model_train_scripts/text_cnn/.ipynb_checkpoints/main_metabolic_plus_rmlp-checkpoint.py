import os
import argparse
import datetime
import torch
# import torchtext.data as data
# import torchtext.datasets as datasets
import model
import train
# import mydatasets

import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
from sklearn.metrics import matthews_corrcoef

import numpy as np


dirs_dct = dict(list(pd.read_csv('../../directory_paths.csv')['paths'].apply(eval)))

checkpoints_dir=dirs_dct['checkpoints_dir']


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='METABOLIC_halfsplit', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=False, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


import os
import argparse
import datetime
import torch
# import torchtext.data as data
# import torchtext.datasets as datasets
import model
import train
# import mydatasets

import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
from sklearn.metrics import matthews_corrcoef

import numpy as np

   
# default_yoon_args = {'lr': 0.001, 'epochs': 256, 'batch_size': 64, 'log_interval': 1, 'test_interval': 100,
# 'save_interval': 500, 'save_dir': 'snapshot', 'early_stop': False, 'save_best': True,
# 'shuffle': False, 'dropout': 0.5, 'max_norm': 3.0, 'embed_dim': 128, 'kernel_num': 100, 'kernel_sizes': [3, 4, 5],
# 'static': False, 'device': -1, 'snapshot': None, 'predict': None, 'test': False, 'embed_num': 30522, 'class_num': 2,
# 'cuda': True, 'no_cuda':False}


# args =argparse.Namespace(**default_yoon_args) 

# load data
print("\nLoading data...")
# Input data (X) should be a list of docs and (y) list of labels
train_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_train_plus_structured.csv'))
val_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_val_plus_structured.csv'))
test_df = pd.read_csv(os.path.join(dirs_dct['data_dir'],'metabolic_test_plus_structured.csv'))

# Split train & val
text_train = train_df['TEXT'] 
text_val =  val_df['TEXT'] 
text_test =  test_df['TEXT'] 

y_train = train_df['label'].astype(int)
y_val = val_df['label'].astype(int)
y_test = test_df['label'].astype(int)


#create class weight dict
# class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
# class_weights_d = dict(enumerate(class_weights))



class_weights=torch.FloatTensor([1,len(y_train)/sum(y_train)])

def bert_encode(data):
    encoded =  bert_tokenizer.encode_plus(
            text=data,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True,
            truncation=True,
            max_length=512
            )
    encoded=encoded['input_ids']
    encoded = encoded 
    encoded = encoded + [bert_tokenizer.pad_token_id]*(512-len(encoded))
    return np.asarray(encoded)

X_train = text_train.apply(bert_encode)
X_val = text_val.apply(bert_encode)
X_test = text_test.apply(bert_encode)

def get_structured(row):
    cols = list(row[[x for x in row.index if 'MED' in x or 'dx' in x or 'min' in x or 'max' in x
                                        or 'mean' in x or 'stdev' in x or 'median' in x or 'count' in x or 'GENDER' in x or 'AGE' in x]])
    return [float(x) for x in cols]
X_train_structured = train_df.apply(get_structured,axis=1)
X_val_structured = val_df.apply(get_structured,axis=1)
X_test_structured = test_df.apply(get_structured,axis=1)


train_inputs = np.stack(X_train.values)
val_inputs = np.stack(X_val.values)
test_inputs = np.stack(X_test.values)

train_dataset = list(zip(train_inputs,np.stack(X_train_structured.values), train_df['label'].astype(int)))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


val_dataset = list(zip(val_inputs,np.stack(X_val_structured.values), val_df['label'].astype(int)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


test_dataset = list(zip(test_inputs,np.stack(X_test_structured.values), test_df['label'].astype(int)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)

[x for x in test_loader]

# update args and print
args.embed_num = len(bert_tokenizer.vocab)
args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

args.pretrained_residual=None
args.num_structured_features=len(X_train_structured.iloc[0])
args.num_res_layers = 10

print(args.__dict__)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text_Plus_Residual(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, None, None, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_loader, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_loader, val_loader,test_loader, cnn, class_weights, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')





