import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from gla_mlp import GLAMLP
from gla_attention import GLAAttention

class GLA(nn.Module):
    def __init__(self, c=5):
        super().__init__()
        gpt2_lmhead = GPT2LMHeadModel.from_pretrained('gpt2')
        self.lm_head = gpt2_lmhead.lm_head
        self.config = gpt2_lmhead.config
        self.wte = gpt2_lmhead.transformer.wte
        self.wpe = gpt2_lmhead.transformer.wpe
        self.drop = gpt2_lmhead.transformer.drop
        self.ln_f = gpt2_lmhead.transformer.ln_f
        
        self.gpt2_layers = []
        for i in range(12):
            tmp = gpt2_lmhead.transformer.h[i]
            tmp.attn = GLAAttention()
            tmp.mlp = GLAMLP()
            self.gpt2_layers.append(tmp)
    def layers(self):
        return self.gpt2_layers, self.config

    def forward(self, X):
        X_int =X
        position_ids = torch.arange(0, X_int.shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)
        X = self.wte(X_int)
        X_p = self.wpe(position_ids)
        X+=X_p
        X = self.drop(X)
        for el in self.gpt2_layers:
            X_init = X
            X = el.ln_1(X)
            X = el.attn(X)
            X = el.ln_2(X[0])
            X = el.mlp(X_init, X)
        X = self.ln_f(X)
        X = self.lm_head(X)
        argmax_indices = torch.argmax(X, dim=2)
        return X,argmax_indices  