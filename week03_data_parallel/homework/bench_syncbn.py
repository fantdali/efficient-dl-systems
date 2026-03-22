import random
from time import perf_counter

import torch
import torch.distributed as dist


NUM_WORKERS = 2
HID_DIMS = [512, 1024]
BATCH_SIZES = [32, 64]


def _bench_worker(rank, world_size, hid_dim, batch_size, impl, device_type, return_dict, port):
    backend = "nccl" if device_type == "cuda" else "gloo"

    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(0)

    local_bs = batch_size // world_size
    x_local = torch.randn(local_bs, hid_dim, device=device, requires_grad=True)

    if impl == "torch":
        bn = torch.nn.SyncBatchNorm(hid_dim, affine=False).to(device)
    else:
        from syncbn import SyncBatchNorm
        bn = SyncBatchNorm(hid_dim).to(device)

    bn.train()

    # Warmup
    for _ in range(5):
        out = bn(x_local)
        loss = out.sum()
        loss.backward()
        if x_local.grad is not None:
            x_local.grad.zero_()

    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    n_iters = 25
    start = perf_counter()

    for _ in range(n_iters):
        out = bn(x_local)
        loss = out.sum()
        loss.backward()
        if x_local.grad is not None:
            x_local.grad.zero_()

    if device_type == "cuda":
        torch.cuda.synchronize()

    time_ms = (perf_counter() - start) * 1000 / n_iters

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
    manager = ctx.Manager()
    return_dict = manager.dict()

    port = random.randint(20000, 40000)

    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=_bench_worker,
            args=(rank, num_workers, hid_dim, batch_size, impl, device_type, return_dict, port),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    return return_dict["stats"]


if __name__ == "__main__":
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_type}, Workers: {NUM_WORKERS}\n")

    print(f"{'impl':<8} | {'hid_dim':>7} | {'batch':>5} | {'time_ms':>8} | {'mem_MB':>8}")
    print("-" * 50)

    for impl in ["torch", "custom"]:
        for hid_dim in HID_DIMS:
            for batch_size in BATCH_SIZES:
                t, m = run_bench(NUM_WORKERS, hid_dim, batch_size, impl, device_type)
                print(f"{impl:<8} | {hid_dim:>7} | {batch_size:>5} | {t:>8.3f} | {m / 1e6:>8.1f}")