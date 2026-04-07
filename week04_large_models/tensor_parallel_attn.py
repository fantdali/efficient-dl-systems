from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


class ComputeWithAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tp_shard: nn.Module, input: torch.Tensor, kwargs):
        input = input.detach().requires_grad_(input.requires_grad)
        ctx.save_for_backward(input)
        ctx._kwargs = kwargs
        ctx._tp_shard = tp_shard
        attn_output, attn_weights = tp_shard(input, **kwargs)
        dist.all_reduce(attn_output)
        return attn_output, attn_weights

    @staticmethod
    def backward(ctx, *grad_output):
        grad_out = grad_output[0]
        with torch.enable_grad():
            output = ctx._tp_shard(ctx.saved_tensors[0], **ctx._kwargs)
            torch.autograd.backward(output[0], grad_out)
        dist.all_reduce(ctx.saved_tensors[0].grad)
        return None, ctx.saved_tensors[0].grad, None


class TPAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]  # (b, s)
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)  # (b, s, heads, d)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.config.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AllReduceAttention(TPAttention):
    def forward(self, input: torch.Tensor, **kwargs):
        return ComputeWithAllReduce.apply(super().forward, input, kwargs)


if __name__ == "__main__":
    dist.init_process_group("gloo")
    torch.manual_seed(1337)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    MODEL_NAME = "unsloth/Llama-3.2-1B"
    config = LlamaConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "sdpa"
    rotary_emb = LlamaRotaryEmbedding(config)

    for active_rank in range(world_size):
        dist.barrier()
        if rank != active_rank:
            continue

        ref_module = LlamaAttention(config, layer_idx=5)

        input = torch.randn(1, 128, config.hidden_size, requires_grad=True)
        position_embeddings = rotary_emb(input, position_ids=torch.arange(128)[None])

        ref_output, _ = ref_module(
            input, attention_mask=None, position_embeddings=position_embeddings
        )
        ref_output.sum().backward()
        ref_input_grad = input.grad.clone()

        tp_config = config
        tp_config.num_attention_heads //= world_size
        tp_config.num_key_value_heads //= world_size
        tp_module = AllReduceAttention(tp_config, layer_idx=5)

        with torch.no_grad():
            k_start = rank * tp_config.num_key_value_heads * tp_module.head_dim
            k_end = (rank + 1) * tp_config.num_key_value_heads * tp_module.head_dim
            tp_module.k_proj.weight.copy_(ref_module.k_proj.weight[k_start:k_end, :])
            tp_module.v_proj.weight.copy_(ref_module.v_proj.weight[k_start:k_end, :])

            q_start = rank * tp_config.num_attention_heads * tp_module.head_dim
            q_end = (rank + 1) * tp_config.num_attention_heads * tp_module.head_dim
            tp_module.q_proj.weight.copy_(ref_module.q_proj.weight[q_start:q_end, :])

            tp_module.o_proj.weight.copy_(ref_module.o_proj.weight[:, q_start:q_end])

        print(f"Initialized {rank=}", flush=True)
        del ref_module

    dist.barrier()
    tp_input = input.detach().requires_grad_(True)
    tp_position_embeddings = rotary_emb(tp_input, position_ids=torch.arange(128)[None])
    tp_output, _ = tp_module(
        tp_input, position_embeddings=tp_position_embeddings, attention_mask=None
    )
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank:
            continue
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        assert torch.allclose(
            tp_output, ref_output, atol=1e-5
        ), f"output mismatch on {rank=}"

    dist.barrier()
    assert tp_input.grad is None
    tp_output.sum().backward()
    if rank == 0:
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank:
            continue
        print(f"TParallel input grad ({rank=}):", tp_input.grad, flush=True)
        assert torch.allclose(
            tp_input.grad, ref_input_grad, atol=1e-5
        ), f"grad mismatch on {rank=}"

    print(f"All checks passed on {rank=}", flush=True)
