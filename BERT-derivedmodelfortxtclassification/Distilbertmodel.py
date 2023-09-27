from transformers import DistilBertModel, DistilBertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
class DistilBert4TC(DistilBertPreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):

        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output[:, 0, :]) # 使用CLS token进行分类

        if labels is not None:
            loss_fact = CrossEntropyLoss()
            preds = logits.view(-1, self.num_labels)
            targs = labels.view(-1)
            loss = loss_fact(preds, targs)
            return loss
        return logits