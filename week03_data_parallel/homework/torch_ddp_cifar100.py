import os
import contextlib
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

torch.set_num_threads(1)

GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = nn.BatchNorm1d(128, affine=False)  # converted to SyncBatchNorm below

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc2(x)


def run_validation(model, val_dataset, device, rank, world_size):
    model.eval()
    sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(val_dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    correct = torch.tensor(0, dtype=torch.float64, device=device)
    total = torch.tensor(0, dtype=torch.float64, device=device)
    val_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction="sum").detach()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().detach()
            total += target.size(0)

    stats = torch.tensor([correct, total, val_loss], dtype=torch.float64, device=device)
    dist.reduce(stats, dst=0, op=dist.ReduceOp.SUM)
    return stats


def run_training(rank, size):
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= size
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if rank == 0:
        train_dataset = CIFAR100("./cifar", train=True, transform=transform, download=True)
        val_dataset = CIFAR100("./cifar", train=False, transform=transform, download=True)
    
    dist.barrier()
    if rank != 0:
        train_dataset = CIFAR100("./cifar", train=True, transform=transform, download=False)
        val_dataset = CIFAR100("./cifar", train=False, transform=transform, download=False)

    train_sampler = DistributedSampler(train_dataset, size, rank)
    loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    model = Net()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if use_cuda:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        epoch_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
        epoch_correct = torch.tensor(0, dtype=torch.float64, device=device)
        epoch_total = torch.tensor(0, dtype=torch.float64, device=device)

        optimizer.zero_grad()
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            # Use no_sync() to skip gradient all-reduce on non-sync steps
            is_sync_step = ((i + 1) % GRAD_ACCUM_STEPS == 0) or (i == len(loader) - 1)
            ctx = contextlib.nullcontext() if is_sync_step else model.no_sync()

            with ctx:
                output = model(data)
                loss = F.cross_entropy(output, target) / GRAD_ACCUM_STEPS
                loss.backward()

            epoch_loss += loss.detach() * GRAD_ACCUM_STEPS
            epoch_correct += (output.argmax(dim=1) == target).sum().detach()
            epoch_total += target.size(0)

            if is_sync_step:
                optimizer.step()
                optimizer.zero_grad()

        # Aggregate training metrics to rank 0
        train_stats = torch.tensor(
            [epoch_loss, epoch_correct, epoch_total], dtype=torch.float64, device=device
        )
        dist.reduce(train_stats, dst=0, op=dist.ReduceOp.SUM)

        # Distributed validation
        val_stats = run_validation(model, val_dataset, device, rank, size)

        if use_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        # Only rank 0 logs
        if rank == 0:
            avg_loss = train_stats[0].item() / train_stats[2].item()
            train_acc = train_stats[1].item() / train_stats[2].item()
            val_acc = val_stats[0].item() / val_stats[1].item()
            val_loss = val_stats[2].item() / val_stats[1].item()
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

    total_time = time.perf_counter() - total_start
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if use_cuda else 0.0
        print(f"\n=== Torch DDP Benchmark Summary ===")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Avg epoch time:      {sum(epoch_times)/len(epoch_times):.2f}s")
        print(f"Peak GPU memory:     {peak_mem:.1f} MB")
        print(f"Final val_acc:       {val_acc:.4f}")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    init_process(local_rank, fn=run_training, backend=backend)
