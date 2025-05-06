import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import os

files = os.listdir('../LCPLM/')
print(files)

# Load the model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("../LCPLM/", trust_remote_code=True)
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

# Retrieve the embeddings
last_hidden_state = outputs.hidden_states[-1]
print(last_hidden_state.shape) # [batch_size, sequence_length, hidden_dim]