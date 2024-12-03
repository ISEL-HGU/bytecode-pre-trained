import torch
import torch.nn as nn
import numpy as np
import random

def split_last(x, shape):
    """Split the last dimension into the given shape."""
    if isinstance(shape, int):
        shape = [shape]
    else:
        shape = list(shape)
    new_shape = x.size()[:-1] + tuple(shape)
    return x.view(*new_shape)

def merge_last(x, n_dims):
    """Merge the last n_dims into a single dimension."""
    if n_dims <= 0 or n_dims > x.dim():
        raise ValueError(f"n_dims must be between 1 and the number of tensor dimensions ({x.dim()})")
    s = x.size()
    new_shape = s[:-n_dims] + (-1,)
    return x.view(*new_shape)


def create_src_mask(src_input_ids, pad_token_id): # for encoder / generate source mask
    src_mask = (src_input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    # (batch_size, 1, 1, src_seq_len)
    return src_mask

def create_tgt_mask(tgt_input_ids, pad_token_id): # for decoder / generate target mask
    tgt_mask = (tgt_input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    seq_len = tgt_input_ids.size(1)
    subsequent_mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=tgt_input_ids.device), diagonal=1).bool()
    tgt_mask = tgt_mask & ~subsequent_mask
    # (batch_size, 1, tgt_seq_len, tgt_seq_len)
    return tgt_mask

# def compute_loss(lm_logits, tgt_input_ids, pad_token_id):
#     shift_logits = lm_logits[:, :-1, :].contiguous()
#     shift_labels = tgt_input_ids[:, 1:].contiguous()

#     loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     return loss

def compute_loss(lm_logits, labels):
    
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss

def set_seeds(seed):
    """ Set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """ get device (CPU or GPU) """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device