import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

class WikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Фильтрация пустых текстов
        self.data = [item for item in dataset if item['text'].strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',  # Дополнение до max_length
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze(0) 
        return input_ids