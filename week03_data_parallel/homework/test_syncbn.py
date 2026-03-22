import random

import torch
import torch.distributed as dist
from syncbn import SyncBatchNorm
import pytest


def _worker(rank, num_workers, hid_dim, batch_size, return_dict, port):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=num_workers,
    )

    torch.manual_seed(42)
    x_full = torch.randn(batch_size, hid_dim)

    local_bs = batch_size // num_workers
    start = rank * local_bs
    end = start + local_bs

    x_local = x_full[start:end].clone().detach().requires_grad_()

    bn = SyncBatchNorm(hid_dim)
    bn.train()

    y_local = bn(x_local)

    # Use a weight mask so all workers always call backward through BN
    half_b = batch_size // 2
    n_contrib = max(0, min(end, half_b) - start)

    weight = torch.zeros(local_bs, 1)
    weight[:n_contrib] = 1.0
    loss = (y_local * weight).sum()
    loss.backward()

    # Gather all outputs and gradients from all workers
    y_list = [torch.zeros_like(y_local) for _ in range(num_workers)]
    dist.all_gather(y_list, y_local)
    y_full = torch.cat(y_list, dim=0)

    grad_list = [torch.zeros_like(x_local) for _ in range(num_workers)]
    dist.all_gather(grad_list, x_local.grad)
    grad_full = torch.cat(grad_list, dim=0)

    if rank == 0:
        return_dict["y_sync"] = y_full.detach()
        return_dict["grad_sync"] = grad_full.detach()

    dist.destroy_process_group()


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()

    port = random.randint(20000, 30000)

    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=_worker,
            args=(rank, num_workers, hid_dim, batch_size, return_dict, port),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

    torch.manual_seed(42)
    x = torch.randn(batch_size, hid_dim, requires_grad=True)

    bn = torch.nn.BatchNorm1d(hid_dim, affine=False, track_running_stats=False)
    bn.train()

    y = bn(x)
    half_b = batch_size // 2
    loss = y[:half_b].sum()
    loss.backward()

    y_sync = return_dict["y_sync"]
    grad_sync = return_dict["grad_sync"]

    assert torch.allclose(y, y_sync, atol=1e-5, rtol=0), \
        f"Forward max diff = {(y - y_sync).abs().max().item()}"
    assert torch.allclose(x.grad, grad_sync, atol=1e-5, rtol=0), \
        f"Backward max diff = {(x.grad - grad_sync).abs().max().item()}"