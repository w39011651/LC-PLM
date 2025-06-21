def ATPtokenize(tokenizer, protein_sequence):
    """
    args:
    tokenizer: esm tokenizer(ex: tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"))
    protein_sequence: sequence from fasta file, (ex: MVVLRQLRLLLWKNYTLKKRKV... in
    >|A0A0G2K1Q8|ABCA3_RAT Phospholipid-transporting ATPase ABCA3 OS=Rattus norvegicus OX=10116 GN=Abca3 PE=1 SV=1
    MVVLRQLRLLLWKNYTLKKRKV...
    )
    """
    encoding = tokenizer(
        protein_sequence,
        return_tensors = 'pt'
    )
    return{'input_ids':encoding['input_ids'], 'attention_mask':encoding['attention_mask']}

def preprocess_information(protein_information:str):
    """
    args:
    protein_information: fasta like format
    return:
    The primary accession and the sequence of the protein
    """
    info = protein_information.split('\n')
    primary_accession = info[0].split('|')[1]
    sequence = ''.join(info[1:])
    return primary_accession, sequence