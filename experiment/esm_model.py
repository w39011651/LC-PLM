import torch
from torch import nn
from transformers import AutoModelForMaskedLM, EsmConfig
import numpy as np

class focal_loss(nn.Module):
    def __init__(self, alpha = None, gamma = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        cross_entropy_loss = nn.functional.cross_entropy(logits, labels, reduce='None', ignore_index=-100)#-log(softmax)
        pt = torch.exp(-cross_entropy_loss)#pt = softmax
        if self.alpha is None:
            focal_loss = ((1-pt) ** self.gamma) * cross_entropy_loss #-(1-softmax^gamma*log(softmax))
        else:
            focal_loss = self.alpha*((1-pt) ** self.gamma) * cross_entropy_loss #-(1-softmax^gamma*log(softmax))
        return focal_loss

class EsmForSequenceLabeling(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(EsmForSequenceLabeling, self).__init__()
        self.esm = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

        self.classifier = nn.Linear(self.esm.config.hidden_size, num_labels)
        self.loss_fn = focal_loss(alpha=None, gamma=2.0).to("cuda")  
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.esm(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states = True)
        sequence_output = outputs.hidden_states[-1]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # mask = (labels != -100)
            # logits = logits[mask]
            # labels = labels[mask]
            # 重塑 logits 和標籤以適應 CrossEntropyLoss
            # logits 從 [batch_size, seq_length, num_classes] 變為 [batch_size * seq_length, num_classes]  
            active_logits = logits.view(-1, logits.size(-1))
          # 標籤從 [batch_size, seq_length] 變為 [batch_size * seq_length]
            active_labels = labels.view(-1)           
          
            loss = self.loss_fn(active_logits, active_labels) 

        return loss, logits