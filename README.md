---
license: cc-by-nc-4.0
tags:
- biology
- protein
---
# LC-PLM

LC-PLM is a frontier long-context protein language model based on an alternative protein LM architecture, BiMamba-S, built off selective structured state-space models. It is pretrained on UniRef50/90 with masked language modeling (MLM) objective. For detailed information on the model architecture, training data, and evaluation performance, please refer to the [accompanying paper](https://arxiv.org/abs/2411.08909).

You can use LC-PLM to extract embeddings for amino acid residues and protein sequences. It can also be fine-tuned to predict residue- or protein- level properties. 

## Getting started

### Install Python dependencies

```bash
pip install transformers mamba-ssm==2.2.2
```

### Download and run inference with the pretrained model

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("amazon/LC-PLM", trust_remote_code=True)
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
```
## Citation

```
@misc{wang2024longcontextproteinlanguagemodel,
      title={Long-context Protein Language Model}, 
      author={Yingheng Wang and Zichen Wang and Gil Sadeh and Luca Zancato and Alessandro Achille and George Karypis and Huzefa Rangwala},
      year={2024},
      eprint={2411.08909},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2411.08909}, 
}
```

## License
This project is licensed under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) License.
