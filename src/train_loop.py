import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


def train_gla(model, train_loader):
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index = 50256)
    
    model.train()
    
    num_epochs = 3
    print("Learning started")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, input_ids in enumerate(train_loader):
            print(batch_idx)
            input_ids = input_ids
    
            inputs = input_ids[:, :-1]  # Все токены, кроме последнего
            targets = input_ids[:, 1:]  # Все токены, кроме первого
    
            # Forward pass
            logits, outputs = model(inputs)
            # Вычисление потерь
            loss = criterion(
        logits.reshape(-1, logits.size(-1)), 
        targets.reshape(-1)                   
    )
            total_loss += loss.item()
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
    
    # Сохранение модели
    torch.save(model.state_dict(), '../models/model_weights.pth')
