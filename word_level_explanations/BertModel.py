import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.file_utils import ModelOutput



class cosine_classifier(nn.Module):
    def __init__(self, config=None, cav=None, layer_number=None):
        super().__init__()
        self.cav=cav,
        self.layer_number=layer_number,
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.cav_tensor = torch.FloatTensor(self.cav*1)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.dist=torch.nn.PairwiseDistance(keepdim=True )
    def forward(self,x):
#         y = self.cos(x.view(1,393216), self.cav_tensor)
        logits=self.cos(x.view(1,393216), self.cav_tensor.to(x.device))
#         logits=y.view(1,1)
#         logits = self.linear(x[:, 0])
#         logits=logits.flatten().view(logits.size())
        logits=logits.add(1)
        logits=logits.divide(2)
        logits = torch.cat((torch.sub(torch.FloatTensor([1]).to(x.device),logits),logits)).view(1,2)
        return logits

class WeightedBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config ,weight_list=None, cav=None, layer_number=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if cav is None:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = cosine_classifier(cav=cav,layer_number=layer_number,config=config)
        self.weight_list = weight_list
        self.cav=cav
        self.layer_number=layer_number
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=True

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.cav is not None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
#             self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
#             self.cav_tensor = torch.tensor([self.cav]*1,device=self.bert.device, requires_grad=True)
#             outputs = self.cos(self.layer_acts.view(1,len(self.cav)), self.cav_tensor)
#             logits=outputs.view(self.layer_acts.size()[0],1)
            pooled_output=outputs.hidden_states[self.layer_number]
        else:    
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
                loss_fct = BCEWithLogitsLoss(weight=self.weight_list)
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.weight_list)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.weight_list)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        if return_logits:
            return logits
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    
class Bert_cutted_cosine(torch.nn.Module):
    def __init__(self, model, bottleneck, cav):
        super(Bert_cutted_cosine, self).__init__()

        layers = [x[1] for x in model.bert.named_modules() if'embeddings' in x[0] and '.' not in x[0]]
        names = [x[0] for x in model.bert.named_modules() if'embeddings' in x[0] and '.' not in x[0]]
        layers = layers+[x[1] for x in model.bert.named_modules() if'layer' in x[0] and x[0].count('.')==2]
        names = names+[x[0] for x in model.bert.named_modules() if'layer' in x[0] and x[0].count('.')==2]

        self.layers = torch.nn.ModuleList()
        self.layers_names = []
        self.bert=model.bert
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.cav=cav
        self.cav_tensor = torch.tensor([self.cav]*1, requires_grad=True)
        self.pooler = model.bert.pooler
        self.dropout=model.dropout
        self.classifier=model.classifier
        self.view1=torch.view
        print(self.bert.device,self.cav_tensor.device)
        bottleneck_met = False
        for name, layer in zip(names, layers):

            if name == bottleneck:

                bottleneck_met = True
                self.layers.append(layer)
                self.layers_names.append(name)
                break  # because we already have the output of the bottleneck layer
            if name == 'AuxLogits':
                continue
            self.layers.append(layer)
            self.layers_names.append(name)
            
    def forward(self, input_ids=None, attention_mask=None):
        y = input_ids

        for i in range(len(self.layers)):
            print(self.layers_names[i])

            y=self.layers[i](y)
            if 'layer' in self.layers_names[i]:
                y=y[-1]
#         y=y.view(1,len(self.cav))
#         y=self.cos(y, self.cav_tensor.to(self.bert.device))
#         print(y.requires_grad)
#         y=y.view(1,1)

#         y = self.pooler(y)
#         y = self.dropout(y)
#         y = self.classifier(y)
#         print(y)
        return y