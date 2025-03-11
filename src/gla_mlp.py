import torch
import torch.nn as nn
from config import hidden_dim, c

class GLAMLP(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, c=c):
        super().__init__()
        self.Wr = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias = False)
    def forward(self, x, o):
        r = nn.SiLU()(self.Wr(x))
        o = self.Wo(r*o)
        return o