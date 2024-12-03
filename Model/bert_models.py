import torch
import torch.nn as nn
import numpy as np
import json
from typing import NamedTuple
import torch.nn.functional as F
from utils import split_last, merge_last, create_src_mask, create_tgt_mask

class Config(NamedTuple):
    vocab_size: int = 30522  
    dim: int = 768
    n_layers: int = 12  
    n_heads: int = 8
    dim_ff: int = 3072
    p_drop_hidden: float = 0.1
    p_drop_attn: float = 0.1
    n_segments: int = 2
    max_len: int = 512

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)  
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, mask):
        h = self.embed(x)  
        for block in self.blocks:
            h = block(h, mask)
        return h  

class EncoderBlock(nn.Module):
    """Transformer 인코더 블록"""
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadedAttention(cfg)
        self.norm1 = LayerNorm(cfg)
        self.ff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
       
        h = self.self_attn(x, x, x, mask)
        h = self.norm1(x + self.drop(h))
        
        h_ff = self.ff(h)
        h = self.norm2(h + self.drop(h_ff))
        return h

class MultiHeadedAttention(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.n_heads = cfg.n_heads
        self.dim_per_head = cfg.dim // cfg.n_heads

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(value)

        
        q = q.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        

        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dim_per_head)
        

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  
            elif mask.dim() == 3:
                mask = mask[:, :, None, :]     
            elif mask.dim() == 4:
                pass 
            else:
                raise ValueError(f"Invalid mask dimension: {mask.dim()}")
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        
        h = torch.matmul(attn, v)  
        h = h.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_per_head)
        

        return h

class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.dim, eps=variance_epsilon)

    def forward(self, x):
        return self.layer_norm(x)

class Embeddings(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)  
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)     
        

        self.norm = nn.LayerNorm(cfg.dim)  
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        
        batch_size, seq_len = x.size()
        
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)  

        
        tok_embedded = self.tok_embed(x)      # (batch_size, seq_len, dim)
        pos_embedded = self.pos_embed(pos)    # (batch_size, seq_len, dim)

        
        embeddings = tok_embedded + pos_embedded  # (batch_size, seq_len, dim)

        
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)

        return embeddings  # (batch_size, seq_len, dim)

class BERT(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.output_projection = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  
        self.pooler = nn.Linear(cfg.dim, cfg.dim)
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_input_ids, src_mask):
        """
        Args:
            src_input_ids: Tensor of shape (batch_size, seq_len)
            src_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            encoded_layers: Tensor of shape (batch_size, seq_len, dim)
            pooled_output: Tensor of shape (batch_size, dim)
            logits: Tensor of shape (batch_size, seq_len, vocab_size) for MLM
        """
        encoded_layers = self.encoder(src_input_ids, src_mask)  # (B, S, D)
        
        cls_hidden = encoded_layers[:, 0, :]  # (B, D)
        pooled_output = self.activation(self.pooler(cls_hidden))  # (B, D)
        
        logits = self.output_projection(encoded_layers)  # (B, S, V)
        return encoded_layers, pooled_output, logits