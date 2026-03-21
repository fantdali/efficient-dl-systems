from functools import reduce
from operator import inv
import re

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm

def _reduce_dims(x: torch.Tensor):
    return (0,) + tuple(range(2, x.dim()))  # reduce over all but C dim

def _broadcast_shape(x: torch.Tensor):
    return (1, x.size(1)) + (1,) * (x.dim() - 2)  # (C,) to (1, C, 1, 1, ...)

class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """
    @staticmethod
    def forward(ctx, x, running_mean, running_var, eps: float, momentum: float, training: bool):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        C = x.size(1)
        reduce_dims = _reduce_dims(x)
        bshape = _broadcast_shape(x)
        
        use_sync = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        if training:
            local_sum = x.sum(dim=reduce_dims)
            local_ssum = (x * x).sum(dim=reduce_dims) 
            local_count_scalar = x.numel() // C
            local_count = torch.full(
                (C,), 
                float(local_count_scalar), 
                device=x.device, 
                dtype=x.dtype
            )
            
            if use_sync:
                packed = torch.cat([local_sum, local_ssum, local_count])
                dist.all_reduce(packed)
                global_sum = packed[:C]
                global_ssum = packed[C:2*C]
                global_count = packed[2*C:]
            else:
                global_sum = local_sum
                global_ssum = local_ssum
                global_count = local_count
            
            mean = global_sum / global_count
            var = (global_ssum / global_count - mean * mean).clamp_min(0.0)
            invstd = torch.rsqrt(var + eps)
            
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
            if running_var is not None:
                running_var.mul_(1 - momentum).add_(momentum * var.detach())

            xhat = (x - mean.view(bshape)) * invstd.view(bshape)
            y = xhat
            
            ctx.save_for_backward(xhat, invstd, global_count)
            ctx.use_sync = use_sync
            ctx.reduce_dims = reduce_dims
            ctx.bshape = bshape

            return y
        
        invstd = torch.rsqrt(running_var + eps)
        y = (x - running_mean.view(bshape)) * invstd.view(bshape)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        y, invstd, global_count = ctx.saved_tensors
        reduce_dims = ctx.reduce_dims
        bshape = ctx.bshape
        
        use_sync = ctx.use_sync
        C = grad_output.size(1)
        
        g = grad_output

        local_sum_dy = g.sum(dim=reduce_dims)
        local_sum_dy_xhat = (g * y).sum(dim=reduce_dims) 
        
        if use_sync:
            packed = torch.cat([local_sum_dy, local_sum_dy_xhat])
            dist.all_reduce(packed)
            global_sum_dy = packed[:C]
            global_sum_dy_xhat = packed[C:2*C]
        else:
            global_sum_dy = local_sum_dy
            global_sum_dy_xhat = local_sum_dy_xhat
        
        t1 = g
        t2 = (global_sum_dy / global_count).view(bshape)
        t3 = y * (global_sum_dy_xhat / global_count).view(bshape)
        grad_input = (t1 - t2 - t3) * invstd.view(bshape)

        return grad_input, None, None, None, None, None
        

class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        return sync_batch_norm.apply(
            input,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self.training,
        )
