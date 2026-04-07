import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class LlamaMLP(nn.Module):  #  based on llama 3.1 8B configuration
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, input):
        return self.down_proj(F.silu(self.gate_proj(input)) * self.up_proj(input))


class ComputeWithAllReduce(torch.autograd.Function):
    @staticmethod  # fun fact: torch.distributed.nn has differentiable all_reduce!
    def forward(ctx, tp_shard: nn.Module, input: torch.Tensor):
        input = input.detach().requires_grad_(input.requires_grad)
        ctx.save_for_backward(input)
        ctx._tp_shard = tp_shard
        output = tp_shard(input)
        dist.all_reduce(output)
        return output
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with torch.enable_grad():
          output = ctx._tp_shard(ctx.saved_tensors[0])
          output.backward(grad_output)
        dist.all_reduce(ctx.saved_tensors[0].grad)
        return None, ctx.saved_tensors[0].grad


class AllReduceModule(nn.Sequential):
    def forward(self, input: torch.Tensor):
        return ComputeWithAllReduce.apply(super().forward, input)


if __name__ == "__main__":
    dist.init_process_group("gloo")   # use nccl for cuda devices
    torch.manual_seed(1337)           # init weights equally on all ranks
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for active_rank in range(world_size):
      dist.barrier()  # initialize each rank sequentially to save system RAM
      if rank != active_rank: continue

      # we will now implement Tensor Parallelism for the ref_module below:
      ref_module = nn.Sequential(nn.RMSNorm(4096), LlamaMLP())
      # compute reference tensors to test against them later
      input = torch.randn(1, 4096, requires_grad=True)
      ref_output = ref_module(input)
      ref_output.sum().backward()
      ref_input_grad = input.grad.clone()

      # TP step 1: define a module that computes a portion of intermediate units
      intermediate_size = ref_module[1].down_proj.in_features
      local_units = intermediate_size // world_size
      assert intermediate_size % world_size == 0
      tp_module = nn.Sequential(   # assign a portion of units per rank --v
          nn.RMSNorm(4096), AllReduceModule(LlamaMLP(intermediate_size=local_units))
      )   # all-reduce outputs during forward, all-reduce gradients on backward

      with torch.no_grad():  # copy select weights from the reference MLP
        # v-- input norm layer is too small to bother parallelizing - we replicate it!
        tp_module[0].load_state_dict(ref_module[0].state_dict())
        # up and gate projections are sharded across output units
        unit_slice = slice(local_units * rank, local_units * (rank + 1))
        tp_module[1][0].up_proj.weight[...] = ref_module[1].up_proj.weight[unit_slice]
        tp_module[1][0].gate_proj.weight[...] = ref_module[1].gate_proj.weight[unit_slice]
        # down projection is sharded across input units, matching up/gate proj
        tp_module[1][0].down_proj.weight[...] = ref_module[1].down_proj.weight[:, unit_slice]
      print(f"Initialized {rank=}", flush=True)
      del ref_module  # free RAM for next rank

    dist.barrier()  # test 1: forward pass
    tp_input = input.detach().requires_grad_(True)
    tp_output = tp_module(tp_input)
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        assert torch.allclose(tp_output, ref_output, atol=1e-6), f"output mismatch on {rank=}"

    dist.barrier()  # test 2: backward w.r.t. inputs
    assert tp_input.grad is None
    tp_output.sum().backward()
    if rank == 0:
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel input grad ({rank=}):", tp_input.grad.data, flush=True)
        assert torch.allclose(tp_input.grad, ref_input_grad, atol=1e-6), f"input_grad mismatch on {rank=}"
