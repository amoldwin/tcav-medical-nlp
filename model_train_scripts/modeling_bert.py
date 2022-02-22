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
import residual
from transformers.file_utils import ModelOutput

class WeightedBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config ,weight_list=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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

        if return_pooled:
            hidden = outputs[1]
        else: 
            hidden = outputs.hidden_states
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden,
            attentions=outputs.attentions,
        )

class WeightedResidualNetwork(BertPreTrainedModel):
    def __init__(self, config ,weight_list=None, num_structured_features=None, return_pooled_output_as_hidden_state=False, num_res_layers=10):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.num_structured_features = num_structured_features
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.weight_list = weight_list
#         self.structured_embed = nn.Linear(num_structured_features, config.hidden_size)
        self.residual = nn.ModuleList([residual.ResidualMLP(num_structured_features, f=torch.nn.functional.tanh) for _ in range(num_res_layers)])
        #second residual mlps for concatenated outputs
#         self.final_residual = nn.ModuleList([residual.ResidualMLP(num_structured_features, f=torch.nn.functional.tanh) for _ in range(num_res_layers)])
        self.final_dense =  nn.Linear(num_structured_features, num_structured_features)
        self.classifier = nn.Linear(num_structured_features, config.num_labels)
        self.return_pooled_output_as_hidden_state= return_pooled_output_as_hidden_state

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
        structured_features=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         structured_features = self.structured_embed(structured_features)

        rhn_outputs = structured_features
        for l in self.residual:
            rhn_outputs = l(structured_features)

        #final residual cells for concatenated outputs
        pooled_output = rhn_outputs
#         for l in self.final_residual:
#             pooled_output = l(pooled_output)
            
        pooled_output = self.final_dense(pooled_output)
        
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
            output = (logits,) + rhn_outputs
            return ((loss,) + output) if loss is not None else output
# extend sequenceclassifieroutput with variable for pooled output
        if self.return_pooled_output_as_hidden_state:
            hs = pooled_output
        else:
            hs = pooled_output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hs,
#             pooled_output=pooled_output
        )



class WeightedBertForSequenceClassificationWithTwinResidualNetwork(BertPreTrainedModel):
    def __init__(self, config ,weight_list=None, num_structured_features=None, return_pooled_output_as_hidden_state=False, num_res_layers=10, pretrained_residual=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.num_structured_features = num_structured_features
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.weight_list = weight_list
#         self.structured_embed = nn.Linear(num_structured_features, config.hidden_size)
        if pretrained_residual==None:
            self.residual = nn.ModuleList([residual.ResidualMLP(num_structured_features, f=torch.nn.functional.tanh) for _ in range(num_res_layers)])
        else:
            just_residual = WeightedResidualNetwork.from_pretrained(pretrained_residual, num_res_layers=10, num_structured_features=88)
            self.residual = just_residual.residual
        #second residual mlps for concatenated outputs
        self.final_residual = nn.ModuleList([residual.ResidualMLP(config.hidden_size+num_structured_features, f=torch.nn.functional.tanh) for _ in range(num_res_layers)])
        self.final_dense =  nn.Linear(config.hidden_size+num_structured_features, config.hidden_size+num_structured_features)
        self.classifier = nn.Linear(config.hidden_size+num_structured_features, config.num_labels)
        self.return_pooled_output_as_hidden_state= return_pooled_output_as_hidden_state

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
        structured_features=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         structured_features = self.structured_embed(structured_features)

        rhn_outputs = structured_features
        for l in self.residual:
            rhn_outputs = l(structured_features)
            
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
        
        #final residual cells for concatenated outputs
        pooled_output = torch.cat((pooled_output, rhn_outputs), dim=-1)
        for l in self.final_residual:
            pooled_output = l(pooled_output)
            
        pooled_output = self.final_dense(pooled_output)
        
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
# extend sequenceclassifieroutput with variable for pooled output
        if self.return_pooled_output_as_hidden_state:
            hs = pooled_output
        else:
            hs = outputs.hidden_states
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hs,
            attentions=outputs.attentions,
#             pooled_output=pooled_output
        )
    
