import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class LCPLMforSequenceLabeling(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super(LCPLMforSequenceLabeling, self).__init__()
        self.lc_plm = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        self.classifier = nn.Linear(self.lc_plm.config.d_model, num_labels)
        #this classifier to determine this amino acid is fad binding site or not
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lc_plm(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states = True)
        sequence_output = outputs.hidden_states[-1]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 重塑 logits 和標籤以適應 CrossEntropyLoss  
            # logits 從 [batch_size, seq_length, num_classes] 變為 [batch_size * seq_length, num_classes]  
            active_logits = logits.view(-1, logits.size(-1))  
          
          # 標籤從 [batch_size, seq_length] 變為 [batch_size * seq_length]  
            active_labels = labels.view(-1)  
          
            loss = self.loss_fn(active_logits, active_labels) 

        return loss, logits