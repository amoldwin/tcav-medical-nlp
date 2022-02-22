import torch.nn as nn
import torch
import torch.nn.functional as F
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
import residual
from transformers.file_utils import ModelOutput

class MLP(BertPreTrainedModel):
    def __init__(self, config ,weight_list=None):
        super().__init__(config)
           
        V = 30522
        D = 128
        DD = 300
        C = 2   
        Ci=1
        self.num_labels = config.num_labels
        self.config = config
        self.embed = nn.Embedding(V, D)
        self.fc1 = nn.Linear(512*D, DD)
        self.fc2 = nn.Linear(DD, C)

        
        self.weight_list = weight_list
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
        return_pooled=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_shape = input_ids.size()

        batch_size=input_shape[0]
        
        x=input_ids
        x = self.embed(x)  # (N, W, D)
        x = x.view(-1,512*128)
#         x = x.unsqueeze(1)  # (N, Ci, W, D)
        
        x= self.fc1(x)
        x=F.relu(x)
        pooled_output = x
        x=self.fc2(x)
        x=F.relu(x)
        
        
        logits = x

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels  )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.weight_list)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        if return_pooled:
            hidden = pooled_output
        else: 
            hidden = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden,
            attentions=None,
        )
