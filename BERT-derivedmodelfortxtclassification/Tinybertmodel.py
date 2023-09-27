from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
class TinyBert4TC(BertPreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        outputs = self.bert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict)

        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fact = CrossEntropyLoss()
            preds = logits.view(-1, self.num_labels)
            targs = labels.view(-1)
            loss = loss_fact(preds, targs)
            return loss
        return logits