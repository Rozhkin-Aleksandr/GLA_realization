import torch
import torch.nn as nn
from math import sqrt
from config import hidden_dim, c
class GLAAttention(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, c=c):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = c
        self.Q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.K = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.V = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W1 = nn.Parameter(torch.Tensor(hidden_dim, 16))
        self.W2 = nn.Parameter(torch.Tensor(16, hidden_dim))
        self.b = nn.Parameter(torch.Tensor(hidden_dim))
        
        nn.init.xavier_normal_(self.Q)
        nn.init.xavier_normal_(self.K)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.b)

        self.S = torch.zeros(hidden_dim, hidden_dim)
        self.register_buffer('base_mask', torch.tril(torch.ones(c, c)))

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                use_cache=False, output_attentions=False):
        batch_size, seq_len, _ = x.shape
        
        Q = torch.matmul(x, self.Q) 
        K = torch.matmul(x, self.K)
        V = torch.matmul(x, self.V)
        
        num_blocks = seq_len // self.C
        remainder = seq_len % self.C
        
        S = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim)
        outputs = []
        
        for i in range(num_blocks):
            start = i * self.C
            end = (i+1) * self.C
            
            Q_block = Q[:, start:end] 
            K_block = K[:, start:end]
            V_block = V[:, start:end]
            K_block_T = K_block.transpose(-1, -2)  

            attn_scores = torch.matmul(Q_block, K_block_T) 
            attn_scores = attn_scores * self.base_mask
            attn_scores = attn_scores / sqrt(self.hidden_dim)
                        
            S_update = torch.matmul(K_block_T, V_block)  
            alpha = torch.matmul(x, self.W1)
            alpha = torch.matmul(alpha, self.W2) + self.b  
            alpha = torch.sigmoid(alpha) ** (1/16) 
            alpha_avg = alpha.mean(dim=1, keepdim=True) 

            alpha_matrix = alpha_avg.repeat_interleave(768, dim=1)  
            S = S * alpha_matrix + S_update            

            output = torch.matmul(Q_block, S) + torch.matmul(attn_scores, V_block)
            outputs.append(output)
        
        if remainder > 0:
            start = num_blocks * self.C
            Q_remain = Q[:, start:] 
            K_remain = K[:, start:]
            V_remain = V[:, start:]
            
            mask = torch.tril(torch.ones(remainder, remainder))
            attn_scores = torch.matmul(Q_remain, K_remain.mT) * mask
            attn_scores = attn_scores / sqrt(self.hidden_dim)
            
            output = torch.matmul(Q_remain, S) + torch.matmul(attn_scores, V_remain)
            outputs.append(output)
        
        O = torch.cat(outputs, dim=1)
        return (O,)
        