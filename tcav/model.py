import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import pandas as pd

import sys
sys.path.insert(0,'../model_train_scripts/')

import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device",device)


class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):
        if bottleneck_name=='input':
            inputs = torch.Tensor(examples).to(device)
            return inputs.detach().cpu().numpy() 
        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        inputs = torch.Tensor(examples).to(device)
        self.model.eval()
        self.model(inputs)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts


class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object.
    """

    def __init__(self, labels_path, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)

    def label_to_id(self, label):
        return self.labels.index(label)


class Bert_rhn_cutted(torch.nn.Module):
    def __init__(self, model, bottleneck):
        super(Bert_rhn_cutted, self).__init__()

        layers = list(model.modules())[1]._modules['encoder']._modules['layer']
        names = [str(i) for i in range(len(layers))]

        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        self.pooler = model.bert.pooler
        self.dropout=model.dropout
        self.classifier=model.classifier
        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'AuxLogits':
                continue
            self.layers.append(layer)
            self.layers_names.append(name)
    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y=self.layers[i](y)
            if type(y)==tuple:
                    y=y[0]
        y = self.pooler(y)
        y = self.dropout(y)
        y = self.classifier(y)
        return y

    
class Bert_cutted(torch.nn.Module):
    def __init__(self, model, bottleneck):
        super(Bert_cutted, self).__init__()

        layers = list(model.modules())[1]._modules['encoder']._modules['layer']
        names = [str(i) for i in range(len(layers))]

        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        self.pooler = model.bert.pooler
        self.dropout=model.dropout
        self.classifier=model.classifier
    def forward(self, x):
        #We are using pooler output, so start after the pooler
        y = x
        y = self.dropout(y)
        y = self.classifier(y)
        return y




class Logistic_cutted(torch.nn.Module):
    def __init__(self, logistic, bottleneck):
        super(Logistic_cutted, self).__init__()
        names = list(logistic._modules.keys())
        layers = list(logistic.children())
        self.all_layers = layers
        self.bottleneck=bottleneck
        self.layers = torch.nn.ModuleList()
        self.layers_names = []
       

    def forward(self, x):
        
        # x is pre-summation, after multiplication of inputs*weights
        batch=x.shape[0]
        x=x.view(batch,2,-1).sum(axis=2)
        logit = F.log_softmax(x, dim=1)
        return logit

class LogisticWrapper(ModelWrapper):
    def __init__( 
      self,
      model_path, target):
      import logistic_model
      import argparse
      input_shape = [512]
      self.model_name = 'logistic_pytorch'
      self.labels_dict = {'negative':0,target:1}
      from torch import nn, optim
     
    

      input_dim = 30522#len(bert_tokenizer.vocab)
      output_dim = 2

      self.model = logistic_model.BoWClassifier( output_dim, input_dim  )
      self.loss_function = nn.NLLLoss()
      self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
      self.model.load_state_dict(torch.load(model_path))
      self.weights = self.model._modules['linear'].weight.to(device)
      self.bias = self.model._modules['linear'].bias.to(device) 
  
  
    def get_cutted_model(self,bottleneck):
        return Logistic_cutted(self.model, bottleneck)

    def label_to_id(self, label):
        return self.labels_dict[label]
    def run_examples(self, examples, bottleneck_name):
        
        
        
        inputs = torch.LongTensor(examples).to(device)
        batch = inputs.shape[0]
        voc_len=inputs.shape[-1]
        self.model.eval()
        
        acts = (self.weights*(inputs.view(batch,1,voc_len))).view(batch,-1)
        acts = acts.view(batch,2,voc_len)
        acts = torch.cat([acts,self.bias.repeat(batch,1).view(batch,2,1)],axis=-1).view(batch,-1)
        acts=acts.detach().cpu().numpy()
        return acts


class CNN_cutted(torch.nn.Module):
    def __init__(self, cnn, bottleneck):
        super(CNN_cutted, self).__init__()
        names = list(cnn._modules.keys())
        layers = list(cnn.children())
        self.all_layers = layers

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        if self.all_layers[1] in self.layers:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.all_layers[1]]  # [(N, Co, W), ...]*len(Ks)
        if self.all_layers[2] in self.layers:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            x = self.all_layers[2](x)  # (N, len(Ks)*Co)
        if self.all_layers[3] in self.layers:
            logit = self.all_layers[3](x)  # (N, C)
        return logit

class CNNPlusStructured_cutted(torch.nn.Module):
    def __init__(self, cnn, bottleneck):
        super(CNNPlusStructured_cutted, self).__init__()
        names = list(cnn._modules.keys())
        layers = list(cnn.children())
        self.all_layers = layers

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)
    def forward(self, x):
        if self.all_layers[-4] in self.layers:
            logit = self.all_layers[-4](x)  # (N, C)
        return logit    
    
class CnnWrapper(ModelWrapper):
    def __init__( 
      self,
      model_path,target):
      import yoon_cnn_model
      import argparse
      input_shape = [512]
      self.model_name = 'cnn_yoon_pytorch'
      self.labels_dict ={'negative':0,target:1}
    
      default_yoon_args = {'lr': 0.001, 'epochs': 256, 'batch_size': 64, 'log_interval': 1, 'test_interval': 100,
        'save_interval': 500, 'save_dir': 'snapshot/2021-07-20_15-52-08', 'early_stop': False, 'save_best': True,
        'shuffle': False, 'dropout': 0, 'max_norm': 3.0, 'embed_dim': 128, 'kernel_num': 100, 'kernel_sizes': [3, 4, 5],
        'static': False, 'device': -1, 'snapshot': None, 'predict': None, 'test': False, 'embed_num': 30522, 'class_num': 2,
        'cuda': True}

      self.model = yoon_cnn_model.CNN_Text(  argparse.Namespace(**default_yoon_args)  )

      self.model.load_state_dict(torch.load(model_path))


    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        inputs = torch.LongTensor(examples).to(device)#.permute(0, 3, 1, 2).to(device)
        self.model.eval()
        self.model(inputs)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts
    def get_cutted_model(self,bottleneck):
        return CNN_cutted(self.model, bottleneck)

    def label_to_id(self, label):
        return self.labels_dict[label]

    
    

class CnnPlusStructuredWrapper(ModelWrapper):
    def __init__( 
      self,
      model_path, target  ):
      import yoon_cnn_model
      import argparse
      input_shape = [512]
      self.model_name = 'cnn_yoon_plus_rmpl_pytorch'
      self.labels_dict ={'negative':0,target:1}
    
      default_yoon_args = {'lr': 0.001, 'epochs': 100, 'batch_size': 64, 'log_interval': 1, 'test_interval': 100,
                           'save_interval': 500, 'save_dir': 'METABOLIC_halfsplit/2021-09-27_17-56-59', 'early_stop': False,
                           'save_best': True, 'shuffle': False, 'dropout': 0, 'max_norm': 3.0, 'embed_dim': 128,
                           'kernel_num': 100,'kernel_sizes': [3, 4, 5], 'static': False, 'device': -1, 'snapshot': None,
                           'predict': None, 'test': False,'embed_num': 30522, 'class_num': 2, 'cuda': True,
                           'pretrained_residual': None, 'num_structured_features': 111, 'num_res_layers': 10}

      self.model = yoon_cnn_model.CNN_Text_Plus_Residual(  argparse.Namespace(**default_yoon_args)  )

      self.model.load_state_dict(torch.load(model_path))



    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        
        df = pd.DataFrame(examples, columns = ['input_ids','structured_features'])
        df['input_ids']=df['input_ids'].apply(eval)

        input_dct =df.to_dict(orient='left')
        structured_features = torch.Tensor(input_dct['structured_features']).to(device)
        input_ids = torch.LongTensor(input_dct['input_ids']).to(device)

        
        
        self.model.eval()
        self.model(x=input_ids, structured_x = structured_features)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts
    def get_cutted_model(self,bottleneck):
        return CNNPlusStructured_cutted(self.model,bottleneck)

    def label_to_id(self, label):
        return self.labels_dict[label]
    
    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None

        return grads


class BertWrapper(ModelWrapper):

    def import_bert_model(self, saved_path):
        import os
        from transformers import BertConfig, BertTokenizer
        from modeling_bert import WeightedBertForSequenceClassification
        self.model = WeightedBertForSequenceClassification.from_pretrained(saved_path, num_labels=2)

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return Bert_cutted(self.model, bottleneck)

    def __init__(
            self,
            model_path,target):
        input_shape = [512]
        self.import_bert_model(model_path)
        self.model_name = 'BertForSequenceClassification'
        layers = list(self.model.modules())[1]._modules['encoder']._modules['layer']
        names = [str(i) for i in range(len(layers))]
        self.layers_dict = dict(zip(names, layers))
        self.labels_dict = {'negative':0,target:1}

        self.cutted_model = self.get_cutted_model('pool').to(device)

    def run_examples(self, examples, bottleneck_name):
        self.model.to(device)
        self.model.eval()
        inputs = torch.tensor(examples, dtype=int).to(device)
        outputs = self.model(inputs, output_hidden_states=True, return_pooled=True)
        acts = outputs['hidden_states'].detach().cpu().numpy()

        return acts

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)

        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.cutted_model
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # y=[i]

        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]

        grads = grads.detach().cpu().numpy()
        cutted_model = None
        # gc.collect()

        return grads

    def label_to_id(self, label):
        return self.labels_dict[label]


class MLP_cutted(torch.nn.Module):
    def __init__(self, model, bottleneck):
        super(MLP_cutted, self).__init__()

        self.classifier = model.fc2

    def forward(self, x):
        # We are using pooler output, so start after the pooler
        y = x
        y = self.classifier(y)
        y = F.relu(y)
        return y


class MLPWrapper(ModelWrapper):

    def import_bert_model(self, saved_path):
        import os
        from transformers import BertConfig, BertTokenizer
        from MLP_model import MLP
        self.model = MLP.from_pretrained(saved_path, num_labels=2)

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return MLP_cutted(self.model, bottleneck)

    def __init__(
            self,
            model_path,target):
        input_shape = [512]
        self.import_bert_model(model_path)
        self.model_name = 'MLP'
        self.cutted_model = self.get_cutted_model('pool').to(device)
        self.labels_dict = {'negative':0,target:1}

    def run_examples(self, examples, bottleneck_name):
        self.model.to(device)
        # inputs = torch.FloatTensor(examples).to(device)
        self.model.eval()
        inputs = torch.tensor(examples, dtype=int).to(device)
        outputs = self.model(inputs, output_hidden_states=True, return_pooled=True)
        acts = outputs['hidden_states'].detach().cpu().numpy()

        return acts

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)

        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.cutted_model
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # y=[i]
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]

        grads = grads.detach().cpu().numpy()
        cutted_model = None
        # gc.collect()

        return grads

    def label_to_id(self, label):
        return self.labels_dict[label]

class BertRHNWrapper(ModelWrapper):

    def import_bert_model(self,saved_path):
        import os
        from transformers import BertConfig,  BertTokenizer
        from modeling_bert import WeightedBertForSequenceClassificationWithTwinResidualNetwork
        self.model = WeightedBertForSequenceClassificationWithTwinResidualNetwork.from_pretrained(saved_path,
                                                                                                  num_structured_features=111,
                                                                                                  num_labels=2,
                                                                                                  return_pooled_output_as_hidden_state=True)
    def forward(self,x):
        return self.model.forward(x)
    def get_cutted_model(self,bottleneck):
        return Bert_rhn_cutted(self.model, bottleneck)
    def __init__( 
      self,
      model_path,target  ):
      self.import_bert_model(model_path)
      self.model_name = 'BertForSequenceClassification'
      layers=[list(self.model.modules())[-2]]
      names = ['finaldense']
      self.layers_dict = dict(zip(names,layers))
      self.labels_dict ={'negative':0,target:1}
    
      self.cutted_models = {str(i):self.get_cutted_model(str(i)).to(device)   for i in range(12)}  
    def run_examples(self, examples, bottleneck_name):

        self.model.to(device)

        self.model.eval()
        df = pd.DataFrame(examples, columns = ['input_ids','structured_features'])
        df['input_ids']=df['input_ids'].apply(eval)
        input_dct =df.to_dict(orient='left')
        structured_features = torch.Tensor(input_dct['structured_features']).to(device)
        input_ids = torch.LongTensor(input_dct['input_ids']).to(device)
        
        outputs = self.model(input_ids = input_ids,structured_features=structured_features, output_hidden_states=True)
        acts = outputs['hidden_states'].detach().cpu().numpy()


        return acts

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)

        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        outputs = self.model.classifier(inputs)
        
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None

        return grads
    def label_to_id(self, label):
        return self.labels_dict[label]


class JustRHNWrapper(ModelWrapper):

    def import_bert_model(self,saved_path):
        import os
        from transformers import BertConfig,  BertTokenizer
        from modeling_bert import WeightedResidualNetwork
        self.model = WeightedResidualNetwork.from_pretrained(saved_path,
                                                                                                  num_structured_features=111,
                                                                                                  num_labels=2,
                                                                                                  return_pooled_output_as_hidden_state=True)
    def forward(self,x):
        return self.model.forward(x)
    
    def __init__( 
      self,
      model_path,target  ):
      self.import_bert_model(model_path)
      self.model_name = 'WeightedResidualNetwork'
      layers=[list(self.model.modules())[-2]]
      names = ['finaldense']
      self.layers_dict = dict(zip(names,layers))
      self.labels_dict ={'negative':0,target:1}
    
    def run_examples(self, examples, bottleneck_name):


        self.model.to(device)
        self.model.eval()
        df = pd.DataFrame(examples, columns = ['input_ids','structured_features'])
        df['input_ids']=df['input_ids'].apply(eval)
        input_dct =df.to_dict(orient='left')
        structured_features = torch.Tensor(input_dct['structured_features']).to(device)
        input_ids = torch.LongTensor(input_dct['input_ids']).to(device)
        
        outputs = self.model(input_ids = input_ids,structured_features=structured_features, output_hidden_states=True)
        acts = outputs['hidden_states'].detach().cpu().numpy()

        return acts

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)

        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        outputs = self.model.classifier(inputs)
        
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None

        return grads
    def label_to_id(self, label):
        return self.labels_dict[label]