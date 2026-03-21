import test
import torch
from torch.autograd import grad
from syncbn import SyncBatchNorm
import torch.distributed as dist
import pytest

def _worker(rank, num_workers, hid_dim, batch_size, return_dict):
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=rank, world_size=num_workers)

    torch.manual_seed(42)

    x = torch.randn(batch_size, hid_dim, requires_grad=True)

    local_bs = batch_size // num_workers
    start = rank * local_bs
    end = start + local_bs
    
    x_local = x[start:end].clone().detach().requires_grad_()
    
    bn = SyncBatchNorm(hid_dim)
    bn.train()
    
    y_local = bn(x_local)
    loss = y_local.sum()
    loss.backward()
    
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
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(num_workers):
        p = ctx.Process(target=_worker, args=(rank, num_workers, hid_dim, batch_size, return_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    torch.manual_seed(42)
    
    x = torch.randn(batch_size, hid_dim, requires_grad=True)
    
    bn = torch.nn.BatchNorm1d(hid_dim, affine=False, track_running_stats=False)
    bn.train()
    
    y = bn(x)
    loss = y.sum()
    loss.backward()
    
    y_sync = return_dict["y_sync"]
    grad_sync = return_dict["grad_sync"]
    
    assert torch.allclose(y, y_sync, atol=1e-4), f"Outputs do not match: {y} vs {y_sync}"
    assert torch.allclose(x.grad, grad_sync, atol=1e-4), f"Gradients do not match: {x.grad} vs {grad_sync}"