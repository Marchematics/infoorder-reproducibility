import inspect
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import CausalBias, causal_lower_right
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

NAME = 'infoorder_phi_cached_sdpa'


def mask(batch_size, q_length, kv_length, q_offset, kv_offset, mask_function,
         attention_mask, use_vmap=False, local_size=None, allow_is_causal_skip=True, **kwargs):
    ordinary_causal = mask_function is causal_mask_function
    if not ordinary_causal and local_size is not None and local_size >= kv_length and allow_is_causal_skip:
        clauses = inspect.getclosurevars(mask_function).nonlocals.get('mask_functions', ())
        ordinary_causal = (len(clauses) == 2 and clauses[1] is causal_mask_function
                          and inspect.getclosurevars(clauses[0]).nonlocals == {'sliding_window': local_size})
    if (batch_size != 1 or kv_offset != 0 or q_offset != kv_length - q_length
            or use_vmap or not ordinary_causal):
        raise ValueError('Phi causal Flash Attention requires one ordinary unpadded sequence')
    if attention_mask is not None and (attention_mask.ndim != 2 or not bool(attention_mask.all())):
        raise ValueError('Phi causal Flash Attention does not accept padding or custom masks')
    return causal_lower_right(q_length, kv_length)


def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    if query.shape[0] != 1 or query.shape[-2] > key.shape[-2]:
        raise ValueError('Invalid Phi causal sequence or cache shape')
    if not isinstance(attention_mask, CausalBias):
        raise ValueError('Phi Flash Attention requires its explicit causal bias')
    if (attention_mask.seq_len_q != query.shape[-2]
            or attention_mask.seq_len_kv != key.shape[-2]):
        raise ValueError('Phi causal bias differs from the actual query and cache lengths')
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout,
            scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return output.transpose(1, 2).contiguous(), None


def register():
    ALL_ATTENTION_FUNCTIONS.register(NAME, attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, mask)
    return NAME
