from wikidatset import WikiTextDataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import batch_size

def load_and_prepare_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Setting pad_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="../data/raw")

    train_dataset = WikiTextDataset(dataset['train'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = WikiTextDataset(dataset['test'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
