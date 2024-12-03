import torch
import torch.nn as nn
import numpy as np
import json
from typing import NamedTuple
import torch.nn.functional as F
from utils import split_last, merge_last, create_src_mask, create_tgt_mask

class Config(NamedTuple):
    vocab_size: int = 51107  # Adjust according to your tokenizer
    dim: int = 768
    n_layers: int = 6  # You can adjust the number of layers
    n_heads: int = 8
    dim_ff: int = 3072
    p_drop_hidden: float = 0.1
    p_drop_attn: float = 0.1
    n_segments: int = 2  # For encoder; decoder may not use this
    max_len: int = 512
    # class_vec_len: int = 128 # vector length of class embedding from AE

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))



class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])  # 수정됨

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h  # (batch_size, seq_len, dim)

class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadedAttention(cfg)
        self.norm1 = LayerNorm(cfg)
        self.ff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        # 자기-어텐션
        h = self.self_attn(x, x, x, mask)
        h = self.norm1(x + self.drop(h))
        # 피드포워드 네트워크
        h_ff = self.ff(h)
        h = self.norm2(h + self.drop(h_ff))
        return h
    

class MultiHeadedAttention(nn.Module):
    """Multi-Headed Attention for Encoder and Decoder"""
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

        # Split into heads
        q = q.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
        # q, k, v: [batch_size, n_heads, seq_len, dim_per_head]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dim_per_head)
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]

        if mask is not None:
            # Adjust mask dimensions
            if mask.dim() == 2:
                # mask가 [batch_size, seq_len_k]인 경우
                mask = mask[:, None, None, :]  # [batch_size, 1, 1, seq_len_k]
            elif mask.dim() == 3:
                # mask가 [batch_size, 1, seq_len_k]인 경우
                mask = mask[:, :, None, :]     # [batch_size, 1, 1, seq_len_k]
            elif mask.dim() == 4:
                pass  # mask already has correct dimensions
            else:
                raise ValueError("Invalid mask dimension: {}".format(mask.dim()))
            # Apply mask
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        # Combine heads
        h = torch.matmul(attn, v)  # [batch_size, n_heads, seq_len_q, dim_per_head]
        h = h.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_per_head)
        # h: [batch_size, seq_len_q, dim]

        return h
    
class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: active_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))
    
class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.dim, eps=variance_epsilon)

    def forward(self, x):
        return self.layer_norm(x)
    
class Embeddings(nn.Module):
    """The embedding module from word, position and token_type embeddings."""
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S, ) -> (B, S)
        t1 = self.tok_embed(x)
        t2 = self.pos_embed(pos)
        if seg is not None:
            t3 = self.seg_embed(seg)
            e = t1 + t2 + t3
        else:
            e = t1 + t2  # seg_embed를 더하지 않음
        return self.drop(self.norm(e))

class DecoderBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, cfg):
        super().__init__()
        self.masked_attn = MultiHeadedAttention(cfg)
        self.norm1 = LayerNorm(cfg)
        self.cross_attn = MultiHeadedAttention(cfg)
        self.norm2 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm3 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, encoder_output, self_mask, cross_mask):
        # Masked Self-Attention
        h = self.masked_attn(x, x, x, self_mask)
        h = self.norm1(x + self.drop(h))

        # Cross-Attention with Encoder Output
        h_cross = self.cross_attn(h, encoder_output, encoder_output, cross_mask)
        h = self.norm2(h + self.drop(h_cross))

        # Position-wise Feed-Forward Network
        h_ff = self.pwff(h)
        h = self.norm3(h + self.drop(h_ff))
        return h
    

class Decoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, encoder_output, self_mask, cross_mask):
        h = self.embed(x, seg=None)  # Decoder typically doesn't use segment embeddings
        for block in self.blocks:
            h = block(h, encoder_output, self_mask, cross_mask)
        return h  # (batch_size, tgt_seq_len, dim)
    

class T5(nn.Module):
    """CodeT5 Model with Encoder and Decoder"""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_projection = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_input_ids, src_seg_ids, tgt_input_ids, src_mask, tgt_mask):
        encoder_output = self.encoder(src_input_ids, src_seg_ids, src_mask)
        decoder_output = self.decoder(tgt_input_ids, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        return logits
    