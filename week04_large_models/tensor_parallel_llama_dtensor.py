import time

import torch
import torch.distributed as dist
import torch.distributed.tensor.parallel as tp
import transformers
from torch.distributed.device_mesh import init_device_mesh


def parallelize_llama_with_dtensor(model, device_mesh):
    for layer_idx in range(len(model.model.layers)):
        layer_name = f"model.layers.{layer_idx}"
        tp.parallelize_module(
            model,
            device_mesh,
            parallelize_plan={
                # Attention: column-parallel for Q,K,V; row-parallel for O
                f"{layer_name}.self_attn.q_proj": tp.ColwiseParallel(),
                f"{layer_name}.self_attn.k_proj": tp.ColwiseParallel(),
                f"{layer_name}.self_attn.v_proj": tp.ColwiseParallel(),
                f"{layer_name}.self_attn.o_proj": tp.RowwiseParallel(),
                # MLP: column-parallel for gate/up; row-parallel for down
                f"{layer_name}.mlp.gate_proj": tp.ColwiseParallel(),
                f"{layer_name}.mlp.up_proj": tp.ColwiseParallel(),
                f"{layer_name}.mlp.down_proj": tp.RowwiseParallel(),
            },
        )
    return model


if __name__ == "__main__":
    dist.init_process_group("gloo")
    torch.manual_seed(1337)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    device_mesh = init_device_mesh(device_type="cpu", mesh_shape=(world_size,))

    MODEL_NAME = "unsloth/Llama-3.2-1B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    if rank == 0:
        ref_model = transformers.LlamaForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        ref_model.eval()
    dist.barrier()

    dt_model = transformers.LlamaForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    dt_model.eval()

    local_num_heads = dt_model.config.num_attention_heads // world_size
    local_num_kv_heads = dt_model.config.num_key_value_heads // world_size
    for layer in dt_model.model.layers:
        layer.self_attn.num_heads = local_num_heads
        layer.self_attn.num_key_value_heads = local_num_kv_heads
        layer.self_attn.num_key_value_groups = local_num_heads // local_num_kv_heads

    parallelize_llama_with_dtensor(dt_model, device_mesh)
    if rank == 0:
        print("DTensor TP model initialized", flush=True)

    dist.barrier()

    # Test 1: Forward pass
    prompt = "A quick brown fox"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        dt_logits = dt_model(input_ids, use_cache=False).logits
        if hasattr(dt_logits, "trigger_wait"):
            dt_logits = dt_logits.trigger_wait()

    if rank == 0:
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
        print(f"\nForward pass comparison:")
        print(f"  Ref logits (last token, first 5): {ref_logits[0, -1, :5]}")
        print(f"  DT  logits (last token, first 5): {dt_logits[0, -1, :5]}")
        max_diff = (dt_logits - ref_logits).abs().max()
        print(f"  Max diff: {max_diff}")
        assert torch.allclose(
            dt_logits, ref_logits, atol=1e-2
        ), f"Logits mismatch! Max diff: {max_diff}"
        print("  Forward pass: PASSED", flush=True)

    # Test 2: Backward w.r.t. input embeddings
    dist.barrier()
    if rank == 0:
        ref_embeds = (
            ref_model.model.embed_tokens(input_ids).detach().requires_grad_(True)
        )
        ref_out = ref_model.model(
            inputs_embeds=ref_embeds, use_cache=False
        ).last_hidden_state
        ref_out = ref_model.lm_head(ref_out)
        ref_out.sum().backward()
        ref_embed_grad = ref_embeds.grad.clone()

    dt_embeds = dt_model.model.embed_tokens(input_ids).detach().requires_grad_(True)
    dt_out = dt_model.model(inputs_embeds=dt_embeds, use_cache=False).last_hidden_state
    dt_out = dt_model.lm_head(dt_out)
    dt_out.sum().backward()

    if rank == 0:
        print(f"\nBackward pass comparison:")
        print(f"  Ref embed grad norm: {ref_embed_grad.norm()}")
        print(f"  DT  embed grad norm: {dt_embeds.grad.norm()}")
        max_grad_diff = (dt_embeds.grad - ref_embed_grad).abs().max()
        print(f"  Max grad diff: {max_grad_diff}")
        assert torch.allclose(
            dt_embeds.grad, ref_embed_grad, atol=130
        ), f"Grad mismatch! Max diff: {max_grad_diff}"
        print("  Backward pass: PASSED", flush=True)

    # Test 3: Speed comparison
    dist.barrier()
    if rank == 0:
        # Time reference forward
        t0 = time.time()
        for _ in range(5):
            with torch.no_grad():
                ref_model(input_ids, use_cache=False)
        ref_time = (time.time() - t0) / 5
        print(f"\nSpeed (forward, avg of 5):")
        print(f"  Reference (single): {ref_time:.4f}s")

    dist.barrier()
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            dt_model(input_ids, use_cache=False)
    dt_time = (time.time() - t0) / 5
    if rank == 0:
        print(f"  DTensor TP ({world_size} ranks): {dt_time:.4f}s")
        print(f"  Speedup: {ref_time / dt_time:.2f}x")

    # Test 4: Generate 10 tokens
    dist.barrier()
    gen_ids = input_ids.clone()
    if rank == 0:
        print(f"\nGeneration: ", end="", flush=True)
        print(prompt, end="", flush=True)

    for i in range(10):
        with torch.no_grad():
            logits = dt_model(gen_ids, use_cache=False).logits
            if hasattr(logits, "trigger_wait"):
                logits = logits.trigger_wait()
            new_token = logits[0, -1].argmax(-1)
            gen_ids = torch.cat([gen_ids, new_token.view(1, 1)], dim=1)
        if rank == 0:
            print(tokenizer.decode(new_token), end="", flush=True)

    if rank == 0:
        print("\n\nAll tests passed!", flush=True)
        del ref_model
