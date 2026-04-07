import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaRotaryEmbedding, LlamaRMSNorm,
    apply_rotary_pos_emb,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward


class AllToAllSeqToHeads(torch.autograd.Function):
    """Reshard from [B, local_seq, H, D] to [B, full_seq, local_H, D] via all_to_all."""
    @staticmethod
    def forward(ctx, x, world_size):
        ctx.world_size = world_size
        B, local_seq, H, D = x.shape
        # Split along head dim into world_size chunks, then all_to_all
        # Input: each rank has [B, local_seq, H, D]
        # We want each rank to get [B, full_seq, local_H, D]
        assert H % world_size == 0
        local_H = H // world_size

        # Reshape to [B, local_seq, world_size, local_H, D]
        x = x.reshape(B, local_seq, world_size, local_H, D)
        # Permute to [world_size, B, local_seq, local_H, D] for all_to_all
        x = x.permute(2, 0, 1, 3, 4).contiguous()

        output = torch.empty_like(x)
        input_list = list(x.unbind(0))
        output_list = list(output.unbind(0))
        dist.all_to_all(output_list, input_list)
        output = torch.stack(output_list, dim=0)

        # output: [world_size, B, local_seq, local_H, D]
        # Permute to [B, world_size, local_seq, local_H, D] then reshape to [B, full_seq, local_H, D]
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = output.reshape(B, world_size * local_seq, local_H, D)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return AllToAllHeadsToSeq.apply(grad_output, ctx.world_size), None


class AllToAllHeadsToSeq(torch.autograd.Function):
    """Reshard from [B, full_seq, local_H, D] to [B, local_seq, H, D] via all_to_all."""
    @staticmethod
    def forward(ctx, x, world_size):
        ctx.world_size = world_size
        B, full_seq, local_H, D = x.shape
        local_seq = full_seq // world_size

        # Reshape to [B, world_size, local_seq, local_H, D]
        x = x.reshape(B, world_size, local_seq, local_H, D)
        # Permute to [world_size, B, local_seq, local_H, D]
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        output = torch.empty_like(x)
        input_list = list(x.unbind(0))
        output_list = list(output.unbind(0))
        dist.all_to_all(output_list, input_list)
        output = torch.stack(output_list, dim=0)

        # output: [world_size, B, local_seq, local_H, D]
        # Permute to [B, local_seq, world_size, local_H, D] then reshape to [B, local_seq, H, D]
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        output = output.reshape(B, local_seq, world_size * local_H, D)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return AllToAllSeqToHeads.apply(grad_output, ctx.world_size), None


class SPAttention(nn.Module):
    """Sequence-parallel attention using sdpa_attention_forward."""
    def __init__(self, config, world_size):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.world_size = world_size
        self.local_num_heads = self.num_heads // world_size
        self.local_num_kv_heads = self.num_kv_heads // world_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        """hidden_states: [B, local_seq, hidden_size]"""
        B, local_seq, _ = hidden_states.shape

        # QKV on local tokens -> [B, local_seq, num_heads/num_kv_heads, head_dim]
        q = self.q_proj(hidden_states).view(B, local_seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, local_seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, local_seq, self.num_kv_heads, self.head_dim)

        # all_to_all: [B, local_seq, H, D] -> [B, full_seq, local_H, D]
        q = AllToAllSeqToHeads.apply(q, self.world_size)
        k = AllToAllSeqToHeads.apply(k, self.world_size)
        v = AllToAllSeqToHeads.apply(v, self.world_size)

        # -> [B, local_heads, full_seq, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE on full sequence positions
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # sdpa_attention_forward needs self.num_key_value_groups for GQA repeat_kv
        attn_output, attn_weights = sdpa_attention_forward(
            self, q, k, v, attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=attention_mask is None,
        )
        # attn_output: [B, full_seq, local_heads * head_dim]
        # Reshape to [B, full_seq, local_heads, head_dim] for all_to_all back
        attn_output = attn_output.view(B, -1, self.local_num_heads, self.head_dim)

        # all_to_all back: [B, full_seq, local_H, D] -> [B, local_seq, H, D]
        attn_output = AllToAllHeadsToSeq.apply(attn_output, self.world_size)

        # [B, local_seq, all_heads * head_dim]
        attn_output = attn_output.reshape(B, local_seq, -1)
        return self.o_proj(attn_output)


class SPDecoderLayer(nn.Module):
    def __init__(self, config, world_size):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SPAttention(config, world_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class SPLlamaModel(nn.Module):
    def __init__(self, config, rank, world_size):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleList([
            SPDecoderLayer(config, world_size) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            full_embeds = self.embed_tokens(input_ids)
        else:
            full_embeds = inputs_embeds

        B, full_seq, D = full_embeds.shape
        local_seq = full_seq // self.world_size
        assert full_seq % self.world_size == 0

        # Each rank takes its token shard
        start = self.rank * local_seq
        hidden_states = full_embeds[:, start:start + local_seq, :]

        # RoPE for full sequence
        position_ids = torch.arange(full_seq, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        # Gather all token hidden states for logits
        all_hidden = [torch.zeros_like(hidden_states) for _ in range(self.world_size)]
        dist.all_gather(all_hidden, hidden_states)
        full_hidden = torch.cat(all_hidden, dim=1)
        return self.lm_head(full_hidden)


def load_sp_model(model_name, rank, world_size):
    config = LlamaConfig.from_pretrained(model_name)
    ref_model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    sp_model = SPLlamaModel(config, rank, world_size)

    with torch.no_grad():
        sp_model.embed_tokens.weight.copy_(ref_model.model.embed_tokens.weight)
        sp_model.norm.weight.copy_(ref_model.model.norm.weight)
        sp_model.lm_head.weight.copy_(ref_model.lm_head.weight)

        for layer_idx in range(config.num_hidden_layers):
            ref_layer = ref_model.model.layers[layer_idx]
            sp_layer = sp_model.layers[layer_idx]

            sp_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
            sp_layer.post_attention_layernorm.weight.copy_(ref_layer.post_attention_layernorm.weight)
            sp_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight)
            sp_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight)
            sp_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight)
            sp_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight)
            sp_layer.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight)
            sp_layer.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight)
            sp_layer.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight)

    del ref_model
    return sp_model, config


if __name__ == "__main__":
    dist.init_process_group("gloo")
    torch.manual_seed(1337)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    MODEL_NAME = "unsloth/Llama-3.2-1B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    if rank == 0:
        ref_model = transformers.LlamaForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, attn_implementation="sdpa",
        )
        ref_model.eval()
    dist.barrier()

    for active_rank in range(world_size):
        dist.barrier()
        if rank != active_rank:
            continue
        sp_model, config = load_sp_model(MODEL_NAME, rank, world_size)
        sp_model.eval()
        print(f"Initialized SP model on {rank=}", flush=True)
    dist.barrier()

    # Use sequence divisible by world_size
    prompt = "The quick brown fox jumps over the lazy dog and then runs fast across the wide open field"
    input_ids = tokenizer(prompt, return_tensors='pt')["input_ids"]
    target_len = (input_ids.shape[1] // world_size) * world_size
    if target_len == 0:
        target_len = world_size
    input_ids = input_ids[:, :target_len]

    # Test 1: Forward
    with torch.no_grad():
        sp_logits = sp_model(input_ids=input_ids)

    if rank == 0:
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
        max_diff = (sp_logits - ref_logits).abs().max()
        print(f"\nForward pass:")
        print(f"  Ref logits (last, first 5): {ref_logits[0, -1, :5]}")
        print(f"  SP  logits (last, first 5): {sp_logits[0, -1, :5]}")
        print(f"  Max diff: {max_diff}")
        assert torch.allclose(sp_logits, ref_logits, atol=1e-2), f"Logits mismatch! Max diff: {max_diff}"
        print("  PASSED", flush=True)

    # Test 2: Backward
    dist.barrier()
    if rank == 0:
        ref_embeds = ref_model.model.embed_tokens(input_ids).detach().requires_grad_(True)
        ref_out = ref_model.model(inputs_embeds=ref_embeds, use_cache=False).last_hidden_state
        ref_model.lm_head(ref_out).sum().backward()
        ref_embed_grad = ref_embeds.grad.clone()

    sp_embeds = sp_model.embed_tokens(input_ids).detach().requires_grad_(True)
    sp_model(inputs_embeds=sp_embeds).sum().backward()

    if rank == 0:
        max_grad_diff = (sp_embeds.grad - ref_embed_grad).abs().max()
        print(f"\nBackward pass:")
        print(f"  Ref embed grad norm: {ref_embed_grad.norm():.4f}")
        print(f"  SP  embed grad norm: {sp_embeds.grad.norm():.4f}")
        print(f"  Max grad diff: {max_grad_diff}")
        assert torch.allclose(sp_embeds.grad, ref_embed_grad, atol=5e-1), f"Grad mismatch! Max diff: {max_grad_diff}"
        print("  PASSED", flush=True)
        print("\nAll sequence parallelism forward tests passed!", flush=True)
        del ref_model
