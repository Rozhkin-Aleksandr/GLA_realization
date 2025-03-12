import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import config

def train_gla(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_tokens, eta_min=config.final_lr)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.vocab_size)
    
    model.train()
    
    print("Training started")
    total_tokens_processed = 0
    best_loss = float('inf')  
    best_model_weights = None 
    
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_idx, input_ids in enumerate(train_loader):
            input_ids = input_ids
            
            inputs = input_ids[:, :-1]  # All tokens except the last one
            targets = input_ids[:, 1:]  # All tokens except the first one
            
            logits, outputs = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)
            )
            if batch_idx%10 == 0:
                print(f"Tokens processed: {total_tokens_processed}, Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip)  
            optimizer.step()

            total_tokens_processed += config.batch_size_tokens            
            if total_tokens_processed < config.warmup_tokens:
                lr_scale = min(1.0, total_tokens_processed / config.warmup_tokens)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.initial_lr * lr_scale
                    
            scheduler.step()
            
            # Save the best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_weights = model.state_dict().copy()
            
            
            total_loss += loss
        total_loss/len(train_loader)
        print(f"Epoch: {epoch}, Loss: {total_loss.item()}")
    
    torch.save(best_model_weights, 'best_model_weights.pth')
    print("Training completed. Best model weights saved to 'best_model_weights.pth'")
