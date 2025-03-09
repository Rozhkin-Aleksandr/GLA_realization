import torch
import torch.nn as nn

class GLAAttention(nn.Module):
    def __init__(self, hidden_dim=768, c=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.C = c
        # Инициализация обучаемых параметров
        self.Q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.K = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.V = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W1 = nn.Parameter(torch.Tensor(hidden_dim, 16))
        self.W2 = nn.Parameter(torch.Tensor(16, hidden_dim))
        self.b = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Инициализация параметров
        nn.init.xavier_normal_(self.Q)
        nn.init.xavier_normal_(self.K)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.b)

        

        # Инициализация параметров
        nn.init.xavier_normal_(self.Q)
        nn.init.xavier_normal_(self.K)
        nn.init.xavier_normal_(self.V)
        self.S = torch.zeros(768, 768)
        self.register_buffer('base_mask', torch.tril(torch.ones(c, c)))

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, 
                use_cache=False, output_attentions=False):
        batch_size, seq_len, _ = x.shape
        
        # Проецирование входных данных
        Q = torch.matmul(x, self.Q)  # [batch, seq, hidden]
        K = torch.matmul(x, self.K)
        V = torch.matmul(x, self.V)
        
        # Разделение на блоки
        num_blocks = seq_len // self.C
        remainder = seq_len % self.C
        
        # Основные блоки
        S = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim, 
                       device=x.device)
        outputs = []
        
        for i in range(num_blocks):
            start = i * self.C
            end = (i+1) * self.C
            
            Q_block = Q[:, start:end]  # [batch, C, hidden]
            K_block = K[:, start:end]
            V_block = V[:, start:end]
            K_block_T = K_block.transpose(-1, -2)  # [batch, hidden, C]

            # Вычисление внимания
            attn_scores = torch.matmul(Q_block, K_block_T)  # [batch, C, C]
            attn_scores = attn_scores * self.base_mask
            attn_scores = attn_scores / sqrt(self.hidden_dim)
            
            # Применение масок
            if attention_mask is not None:
                attn_scores += attention_mask[:, start:end, start:end]
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Обновление состояния
            S_update = torch.matmul(K_block_T, V_block)  # [batch, hidden, hidden]
            alpha = torch.matmul(x, self.W1)
            alpha = torch.matmul(alpha, self.W2) + self.b  # [batch, seq, hidden]
            alpha = torch.sigmoid(alpha) ** (1/16)  # Применение сигмоида и возведение в степень
            alpha_avg = alpha.mean(dim=1, keepdim=True)  # [1,1,768]

            # Создаём матрицу через повторение
            alpha_matrix = alpha_avg.repeat_interleave(768, dim=1)  # [1,768,768]
            S = S * alpha_matrix + S_update            
            # Вычисление выхода
            output = torch.matmul(Q_block, S) + torch.matmul(attn_weights, V_block)
            outputs.append(output)
        
        # Обработка остатка
        if remainder > 0:
            start = num_blocks * self.C
            Q_remain = Q[:, start:]  # [batch, rem, hidden]
            K_remain = K[:, start:]
            V_remain = V[:, start:]
            
            mask = torch.tril(torch.ones(remainder, remainder, device=x.device))
            attn_scores = torch.matmul(Q_remain, K_remain.mT) * mask
            attn_scores = attn_scores / sqrt(self.hidden_dim)
            
            if attention_mask is not None:
                attn_scores += attention_mask[:, start:, start:]
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(Q_remain, S) + torch.matmul(attn_weights, V_remain)
            outputs.append(output)
        
        # Сборка выходов
        O = torch.cat(outputs, dim=1)
        return (O,)
        