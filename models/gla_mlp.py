import torch
import torch.nn as nn


class GLAMLP(nn.Module):
    def __init__(self, hidden_dim=768, c=5):
        super().__init__()
        self.Wr = nn.Linear(768, 768, bias = True)
        self.Wo = nn.Linear(768, 768, bias = False)
    def forward(self, x, o):
        r = nn.SiLU()(self.Wr(x))
        o = self.Wo(r*o)
        return o