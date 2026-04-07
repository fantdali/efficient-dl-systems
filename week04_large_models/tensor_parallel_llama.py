import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaRotaryEmbedding, LlamaRMSNorm,
)
from tensor_parallel_attn import AllReduceAttention
from tensor_parallel_mlp import AllReduceModule, LlamaMLP


class TPDecoderLayer(nn.Module):
    def __init__(self, tp_config, local_intermediate, layer_idx):
        super().__init__()
        hidden_size = tp_config.hidden_size
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=tp_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=tp_config.rms_norm_eps)
        self.self_attn = AllReduceAttention(tp_config, layer_idx=layer_idx)
        self.mlp = AllReduceModule(LlamaMLP(hidden_size=hidden_size, intermediate_size=local_intermediate))

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TPLlamaModel(nn.Module):
    def __init__(self, tp_config, full_config, local_intermediate):
        super().__init__()
        hidden_size = full_config.hidden_size
        self.embed_tokens = nn.Embedding(full_config.vocab_size, hidden_size)
        self.norm = LlamaRMSNorm(hidden_size, eps=full_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(full_config)
        self.lm_head = nn.Linear(hidden_size, full_config.vocab_size, bias=False)
        self.layers = nn.ModuleList([
            TPDecoderLayer(tp_config, local_intermediate, layer_idx=i)
            for i in range(full_config.num_hidden_layers)
        ])

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=hidden_states.dtype, device=hidden_states.device),
            diagonal=1,
        )[None, None, :, :]  # (1, 1, seq, seq) 

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask=causal_mask)

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def load_tp_model(model_name, rank, world_size):
    full_config = LlamaConfig.from_pretrained(model_name)
    head_dim = getattr(full_config, "head_dim", full_config.hidden_size // full_config.num_attention_heads)

    tp_config = copy.deepcopy(full_config)
    tp_config.num_attention_heads //= world_size
    tp_config.num_key_value_heads //= world_size
    local_intermediate = full_config.intermediate_size // world_size

    ref_model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tp_model = TPLlamaModel(tp_config, full_config, local_intermediate)

    with torch.no_grad():
        tp_model.embed_tokens.weight.copy_(ref_model.model.embed_tokens.weight)
        tp_model.norm.weight.copy_(ref_model.model.norm.weight)
        tp_model.lm_head.weight.copy_(ref_model.lm_head.weight)

        for layer_idx in range(full_config.num_hidden_layers):
            ref_layer = ref_model.model.layers[layer_idx]
            tp_layer = tp_model.layers[layer_idx]

            tp_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
            tp_layer.post_attention_layernorm.weight.copy_(ref_layer.post_attention_layernorm.weight)

            q_slice = slice(rank * tp_config.num_attention_heads * head_dim,
                            (rank + 1) * tp_config.num_attention_heads * head_dim)
            kv_slice = slice(rank * tp_config.num_key_value_heads * head_dim,
                             (rank + 1) * tp_config.num_key_value_heads * head_dim)

            tp_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight[q_slice])
            tp_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight[kv_slice])
            tp_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight[kv_slice])
            tp_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight[:, q_slice])

            int_slice = slice(rank * local_intermediate, (rank + 1) * local_intermediate)

            tp_layer.mlp[0].gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight[int_slice])
            tp_layer.mlp[0].up_proj.weight.copy_(ref_layer.mlp.up_proj.weight[int_slice])
            tp_layer.mlp[0].down_proj.weight.copy_(ref_layer.mlp.down_proj.weight[:, int_slice])

    del ref_model
    return tp_model, full_config


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
        tp_model, full_config = load_tp_model(MODEL_NAME, rank, world_size)
        tp_model.eval()
        print(f"Initialized TP model on {rank=}", flush=True)

    dist.barrier()

    # Test 1: Forward pass
    prompt = "A quick brown fox"
    input_ids = tokenizer(prompt, return_tensors='pt')["input_ids"]

    with torch.no_grad():
        tp_logits = tp_model(input_ids=input_ids)

    if rank == 0:
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
        print(f"\nForward pass comparison:")
        print(f"  Ref logits (last token, first 5): {ref_logits[0, -1, :5]}")
        print(f"  TP  logits (last token, first 5): {tp_logits[0, -1, :5]}")
        assert torch.allclose(tp_logits, ref_logits, atol=1e-3), \
            f"Logits mismatch! Max diff: {(tp_logits - ref_logits).abs().max()}"
        print("  Forward pass: PASSED", flush=True)

    # Test 2: Backward
    dist.barrier()
    if rank == 0:
        ref_embeds = ref_model.model.embed_tokens(input_ids).detach().requires_grad_(True)
        ref_out = ref_model.model(inputs_embeds=ref_embeds, use_cache=False).last_hidden_state
        ref_model.lm_head(ref_out).sum().backward()
        ref_embed_grad = ref_embeds.grad.clone()

    tp_embeds = tp_model.embed_tokens(input_ids).detach().requires_grad_(True)
    tp_model(inputs_embeds=tp_embeds).sum().backward()

    if rank == 0:
        print(f"\nBackward pass comparison:")
        print(f"  Ref embed grad norm: {ref_embed_grad.norm()}")
        print(f"  TP  embed grad norm: {tp_embeds.grad.norm()}")
        assert torch.allclose(tp_embeds.grad, ref_embed_grad, atol=150), \
            f"Grad mismatch! Max diff: {(tp_embeds.grad - ref_embed_grad).abs().max()}"
        print("  Backward pass: PASSED", flush=True)

    # Test 3: Generate
    dist.barrier()
    gen_ids = input_ids.clone()
    if rank == 0:
        print(f"\nGeneration: {prompt}", end="", flush=True)

    for _ in range(5):
        with torch.no_grad():
            new_token = tp_model(input_ids=gen_ids)[0, -1].argmax(-1)
            gen_ids = torch.cat([gen_ids, new_token.view(1, 1)], dim=1)
        if rank == 0:
            print(tokenizer.decode(new_token), end="", flush=True)

    if rank == 0:
        print("\n\nAll tests passed!", flush=True)
        del ref_model
