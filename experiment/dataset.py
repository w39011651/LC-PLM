import torch
from torch.utils.data import Dataset

class FADBindingDataset(Dataset):
    def __init__(self, tokenizer, sequences, labels, max_length=512):
        """
        for data_list, datalist should have the following structure: 
        [{"sequence:..., "label":...},...]
        """
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            sequence = self.sequences[idx]
            label = self.labels[idx]

            encoding = self.tokenizer(
                sequence,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            padded_label = torch.full((self.max_length,), -100, dtype=torch.long)
            valid_length = min(len(label), self.max_length)
            padded_label[:valid_length] = torch.tensor([int(l) for l in label[:valid_length]]) 

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': padded_label.clone().detach()
            }

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise None  # 或 raise，讓你能看到是哪筆資料報錯
        
from torch.utils.data import random_split

def split_dataset(dataset:FADBindingDataset, split_size = 0.8):
    train_size = int(split_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return [train_dataset, test_dataset]