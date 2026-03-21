import os

import torch.distributed as dist
import torch


def run_sequential(rank, size, num_iter=3):
    """
    Prints the process rank sequentially in two orders over `num_iter` iterations,
    separating the output for each iteration by `---`.
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ---
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ```
    """
    
    t = torch.zeros(1)

    for i in range(num_iter):
        # asc
        if rank == 0:
            print(f"Process {rank}")
            dist.send(t, dst=(rank + 1) % size)
            dist.recv(t, src=(size - 1) % size)
        else:
            dist.recv(t, src=(rank - 1) % size)
            print(f"Process {rank}")
            dist.send(t, dst=(rank + 1) % size)
        
        # desc
        if rank == size-1:
            print(f"Process {rank}")
            dist.send(t, dst=(rank - 1) % size)
            dist.recv(t, src=0)
        else:
            dist.recv(t, src=(rank + 1) % size)
            print(f"Process {rank}")
            dist.send(t, dst=(rank - 1) % size)
        
        if rank == 0 and i < num_iter - 1:
            print("---")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
