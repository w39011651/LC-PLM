import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Any

def get_parameter_number(model:Any):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total":total_num, "Trainable": trainable_num}


# Load the model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("../LCPLM", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Input a protein sequence:
# fun fact: this is [Mambalgin-1](https://www.uniprot.org/uniprotkb/P0DKR6/entry) from Black mamba
sequence = "MKTLLLTLLVVTIVCLDLGYSLKCYQHGKVVTCHRDMKFCYHNTGMPFRNLKLILQGCSSSCSETENNKCCSTDRCNK"

# Tokenize the sequence:
inputs = tokenizer(sequence, return_tensors="pt")

# Inference with LC-PLM on GPU
device = torch.device("cuda:0")
model = model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    print(get_parameter_number(model))

# Retrieve the embeddings
last_hidden_state = outputs.hidden_states[-1]
print(last_hidden_state.shape) # [batch_size, sequence_length, hidden_dim]