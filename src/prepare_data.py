from wikidataset import WikiTextDataset
from transformers import GPT2Tokenizer


def load_and_prepare_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="../data/raw")
    train_dataset = WikiTextDataset(dataset['train'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    return train_loader