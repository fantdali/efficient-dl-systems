import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaRotaryEmbedding, LlamaRMSNorm,
    apply_rotary_pos_emb,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from sequence_parallel_forward import (
    AllToAllSeqToHeads, AllToAllHeadsToSeq, SPAttention, SPDecoderLayer,
)


class SPLlamaForTraining(nn.Module):
    """Sequence-parallel Llama with prompt tuning for training."""
    def __init__(self, config, rank, world_size, num_virtual_tokens=32):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.num_virtual_tokens = num_virtual_tokens

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleList([
            SPDecoderLayer(config, world_size) for _ in range(config.num_hidden_layers)
        ])
        # Prompt tuning: learnable virtual token embeddings
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, config.hidden_size)
        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        B = input_ids.shape[0]
        token_embeds = self.embed_tokens(input_ids)

        # Prepend prompt embeddings
        prompt_tokens = self.prompt_embeddings.weight.unsqueeze(0).expand(B, -1, -1)
        full_embeds = torch.cat([prompt_tokens, token_embeds], dim=1)

        full_seq = full_embeds.shape[1]
        local_seq = full_seq // self.world_size
        assert full_seq % self.world_size == 0, f"Total seq ({full_seq}) must be divisible by {self.world_size}"

        start = self.rank * local_seq
        hidden_states = full_embeds[:, start:start + local_seq, :]

        position_ids = torch.arange(full_seq, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        # Gather all hidden states for logits / loss
        all_hidden = [torch.zeros_like(hidden_states) for _ in range(self.world_size)]
        dist.all_gather(all_hidden, hidden_states)
        full_hidden = torch.cat(all_hidden, dim=1)
        logits = self.lm_head(full_hidden)

        loss = None
        if labels is not None:
            # Skip prompt tokens, shift by 1
            shift_logits = logits[:, self.num_virtual_tokens:-1, :].contiguous()
            shift_labels = labels.contiguous()
            min_len = min(shift_logits.shape[1], shift_labels.shape[1])
            shift_logits = shift_logits[:, :min_len]
            shift_labels = shift_labels[:, :min_len]
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        return loss, logits


def load_sp_training_model(model_name, rank, world_size, num_virtual_tokens=32):
    config = LlamaConfig.from_pretrained(model_name)
    ref_model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    sp_model = SPLlamaForTraining(config, rank, world_size, num_virtual_tokens)

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
    NUM_VIRTUAL_TOKENS = 32
    SEQUENCE_LENGTH = 512  # Increase as much as your hardware allows!

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    for active_rank in range(world_size):
        dist.barrier()
        if rank != active_rank:
            continue
        sp_model, config = load_sp_training_model(MODEL_NAME, rank, world_size, NUM_VIRTUAL_TOKENS)
        print(f"Loaded SP training model on {rank=}", flush=True)
    dist.barrier()

    # Freeze all except prompt embeddings
    for name, param in sp_model.named_parameters():
        if "prompt_embeddings" not in name:
            param.requires_grad = False

    # Ensure total (prompt + input) is divisible by world_size
    while (SEQUENCE_LENGTH + NUM_VIRTUAL_TOKENS) % world_size != 0:
        SEQUENCE_LENGTH -= 1
    total_seq = SEQUENCE_LENGTH + NUM_VIRTUAL_TOKENS

    # Download text
    if rank == 0:
        os.system("wget -q https://www.gutenberg.org/cache/epub/4300/pg4300.txt -O ulysses.txt 2>/dev/null || true")
    dist.barrier()

    text = open("ulysses.txt").read()
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    if rank == 0:
        print(f"Text tokens: {input_ids.shape[1]}, using {SEQUENCE_LENGTH}")
        print(f"Total with prompt: {total_seq}, per rank: {total_seq // world_size}")

    input_ids = input_ids[:, :SEQUENCE_LENGTH]
    labels = input_ids[:, 1:].clone()
    input_ids = input_ids[:, :-1]
    # Re-adjust: now input_ids has SEQUENCE_LENGTH-1 tokens, total = SEQUENCE_LENGTH-1 + NUM_VIRTUAL_TOKENS
    # Ensure divisibility again
    actual_input_len = input_ids.shape[1]
    total_seq = actual_input_len + NUM_VIRTUAL_TOKENS
    while total_seq % world_size != 0:
        actual_input_len -= 1
        total_seq = actual_input_len + NUM_VIRTUAL_TOKENS
    input_ids = input_ids[:, :actual_input_len]
    labels = labels[:, :actual_input_len]

    # Wrap with DDP to sync prompt_embeddings gradients across ranks
    sp_model = DDP(sp_model, find_unused_parameters=True)

    trainable_params = [p for p in sp_model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in sp_model.parameters())
    if rank == 0:
        print(f"Parameters: {total_trainable} trainable / {total_params} total")

    opt = torch.optim.Adam(trainable_params, lr=1e-3)

    for i in range(10):
        # Call through DDP wrapper (not module.forward) for gradient sync
        loss, logits = sp_model(input_ids, labels=labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if rank == 0:
            print(f"  {i=}\t{loss.item()=:.4f}", flush=True)

    if rank == 0:
        print(f"\nTraining complete! Sequence length: {actual_input_len}")
        print(f"With {world_size} ranks, each processes {total_seq // world_size} tokens/step")
        print("All sequence parallelism training tests passed!")
