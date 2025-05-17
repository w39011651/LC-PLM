import json
import torch

def read_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    return data

def sliding_window(protein_info, window_size = 13):
    ret = []
    left = 0
    right = window_size
    length = len(protein_info['sequence'])
    
    while right + 1 != length:#because slice is interval[left:right), so we can slice to idx, where idx is the length of sequence
        subseq_info = {"sequence":None, "label":None}
        subseq_info['sequence'] = protein_info['sequence'][left:right]
        subseq_info['label'] = protein_info['label'][left:right]
        #subseq_info['target_amino_acid'] = protein_info['sequence'][(left+right)//2]
        ret.append(subseq_info)
        left += 1
        right += 1
    
    return ret

def get_binding_site(features):
    binding_site = []
    for feature in features:
        if feature['type'] == 'Binding site' and feature['ligand']['name']=='ATP':#only need FAD
            binding_site.append([feature['location']['start']['value'], feature['location']['end']['value']])
    return binding_site

def generate_label(protein, positions):
    n = protein['sequence']['length']
    labels = [0 for _ in range(n)]

    for bind_site in positions:
        for i in range(bind_site[0], bind_site[1]+1):
            labels[i-1] = 1
    return labels

def get_protein_information(json_data):
    protein_information = []

    for protein in json_data['results']:
        info = {"sequence":None, "label":None}
        info['sequence'] = protein['sequence']['value']
        info['label'] = generate_label(protein, get_binding_site(protein['features']))

        if info['label'].count(0) == len(info['label']):
            continue

        protein_information.append(info)

    return protein_information

def get_aa_info(protein_information):
    aa_information = []
    for protein in protein_information:
        aa_information += sliding_window(protein) 
    return aa_information

MAX_LEN = 512

def tokenize(tokenizer, nlp_pretoken, label_pretoken):
    encoding = tokenizer(nlp_pretoken, 
                         is_split_into_words=True, 
                         add_special_tokens = True,
                         return_tensors="pt", 
                         padding="max_length", 
                         truncation=True, 
                         max_length=512
                         )
    
    word_ids = encoding.word_ids()
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:   
            label_ids.append(label_pretoken[word_idx])
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(label_ids, dtype=float),
    }

def customTokenize(tokenizer, nlp_pretoken, label_pretoken):
    # 將蛋白質序列轉為空格分隔格式（ProtBERT 需要）
    nlp_pretoken = " ".join(nlp_pretoken)
    encoding = tokenizer(
        nlp_pretoken,
        is_split_into_words=False,  # 已經手動分隔
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # 創建標籤，長度與 input_ids 一致
    label_ids = torch.zeros(512, dtype=torch.float32)  # 初始化為 0.0
    # 將原始標籤映射到 token 位置（跳過 [CLS] 和 [SEP]）
    for i, idx in enumerate(range(len(label_pretoken))):
        if i < 510:  # 留空間給 [CLS] 和 [SEP]
            label_ids[i + 1] = float(label_pretoken[idx])  # 從索引 1 開始

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": label_ids,
    }

from torch.utils.data import random_split

from tqdm import tqdm

import numpy as np



def run(file_path ="./FAD_rmsim.json",  slidingwindow=False):
    """RETURN [TRAIN_DATASET, TEST_DATASET]"""
    data = read_from_json(file_path)
    protein_information = get_protein_information(json_data=data)
    
    if slidingwindow:
        ret = []
        for protein in protein_information:
            ret += sliding_window(protein)
        return ret
    
    return protein_information

import os
if __name__ == '__main__':
    files = os.listdir('./')
    print(files)
    data = run('experiment/ATP_rmsim.json', False)
    print(len(data))
