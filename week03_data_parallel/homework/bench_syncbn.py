import dis
import random
from time import perf_counter

import torch
import torch.distributed as dist

def _bench_worker(rank, world_size, hid_dim, batch_size, impl, device_type, return_dict, port):
    backend = "nccl" if device_type == "cuda" else "gloo"

    dist.init_process_group(
        backend, 
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, 
        world_size=world_size
    )
    
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)
    
    x = torch.randn(batch_size, hid_dim, device=device, requires_grad=True)
    
    local_bs = batch_size // world_size
    x_local = x[rank*local_bs:(rank+1)*local_bs].clone().detach().requires_grad_()
    
    if impl == "torch":
        if device_type != "cuda":
            if rank == 0:
                return_dict["stats"] = (0, 0)
        bn = torch.nn.SyncBatchNorm(hid_dim).to(device)
    else:
        from syncbn import SyncBatchNorm
        bn = SyncBatchNorm(hid_dim).to(device)
    
    bn.train()
    
    for _ in range(5):
        out = bn(x_local)
        loss = out.sum()
        loss.backward()
        x_local.grad.zero_()
    
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()

    for _ in range(25):
        out = bn(x_local)
        loss = out.sum()
        loss.backward()
        x_local.grad.zero_()

    if device_type == "cuda":
        torch.cuda.synchronize() 
    
    time_ms = (perf_counter() - start) * 1000 / 25
    if device_type == "cuda":
        mem = torch.cuda.max_memory_allocated(device)
    else:
        mem = 0
    
    if rank == 0:
        return_dict["stats"] = (time_ms, mem)
    
    dist.barrier()
    dist.destroy_process_group()

def run_bench(num_workers, hid_dim, batch_size, impl, device_type="cpu"):
    ctx = torch.multiprocessing.get_context("spawn")
    manageer = ctx.Manager()
    return_dict = manageer.dict()
    
    port = random.randint(20000, 40000)
    
    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=_bench_worker, 
            args=(rank, num_workers, hid_dim, batch_size, impl, device_type, return_dict, port)
        )
        p.start()
        procs.append(p)
    
    for p in procs:
        p.join()
        
    return return_dict["stats"]

if __name__ == "__main__":
    configs = [
        (2, 1024, 8),
        (2, 1024, 64),
        (2, 1024, 128),
    ]
    print("Running on CPU:")
    for impl in ["custom"]:
        for num_workers, hid_dim, batch_size in configs:
            t, m = run_bench(num_workers, hid_dim, batch_size, impl, device_type="cpu")
            print(f"{impl} | workers={num_workers}, batch_size={batch_size} | time={t:.3f} ms | mem={m/1e6:.1f} MB")

    configs = [
        (1, 1024, 8),
        (1, 1024, 64),
        (1, 1024, 128),
    ]
    print("Running on GPU:")
    for impl in ["torch", "custom"]:
        for num_workers, hid_dim, batch_size in configs:
            t, m = run_bench(num_workers, hid_dim, batch_size, impl, device_type="cuda")
            print(f"{impl} | workers={num_workers}, batch_size={batch_size} | time={t:.3f} ms | mem={m/1e6:.1f} MB")