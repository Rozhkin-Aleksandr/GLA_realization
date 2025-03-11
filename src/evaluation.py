import torch
import torch.nn as nn
import math
from config import vocab_size

def calculate_perplexity(loss):
    return math.exp(loss)

def evaluate_perplexity(model, test_loader):
    model.eval()  
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index = vocab_size)
    print("Evaluation started", len(test_loader))
    with torch.no_grad():  
        for batch_idx, input_ids in enumerate(test_loader):
            input_ids = input_ids

            inputs = input_ids[:, :-1]  
            targets = input_ids[:, 1:]  

            logits, _ = model(inputs)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)                   
            )
            total_loss += loss.item() * inputs.size(0)  
            total_tokens += inputs.size(0)  
            if batch_idx %10 ==0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    print(f"Perplexity on test dataset: {perplexity:.4f}")

