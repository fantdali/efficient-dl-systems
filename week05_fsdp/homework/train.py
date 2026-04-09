import functools
import logging
import os
import pathlib
import pickle
from typing import Any, Literal

import torch
import tyro
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule as FSDP2Module
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import fully_shard as fsdp2_fully_shard
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import Transformer

from fsdp import FSDPCommContext
from fsdp import FSDPModule as EffdlFSDPModule
from fsdp import fully_shard as effdl_fully_shard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

STR_TO_TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def apply_fsdp2(
    model: Transformer,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype | None = None,
    reshard_after_forward: bool = True,
) -> None:
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
    }
    fsdp2_fully_shard(
        model.tok_embeddings,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for transformer_block in model.layers.values():
        fsdp2_fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fsdp2_fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    fsdp2_fully_shard(model, **fsdp_config)


def apply_effdl_fsdp(
    model: Transformer,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype | None = None,
    reshard_after_forward: bool = True,
) -> None:
    comm_ctx = FSDPCommContext(dp_mesh.device_type)
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"comm_ctx": comm_ctx, "mesh": dp_mesh, "mp_policy": mp_policy}
    effdl_fully_shard(
        model.tok_embeddings,
        module_fqn="tok_embeddings",
        reshard_after_forward=reshard_after_forward,
        **fsdp_config,
    )
    for layer_name, layer in model.layers.items():
        effdl_fully_shard(
            layer,
            module_fqn=f"layers.{layer_name}",
            reshard_after_forward=reshard_after_forward,
            **fsdp_config,
        )
    effdl_fully_shard(
        model.norm,
        module_fqn="norm",
        reshard_after_forward=reshard_after_forward,
        **fsdp_config,
    )
    effdl_fully_shard(
        model.output,
        module_fqn="output",
        reshard_after_forward=reshard_after_forward,
        **fsdp_config,
    )


def build_dataloader(seq_len: int, batch_size: int = 1) -> StatefulDataLoader:
    dataset = HuggingFaceTextDataset(
        dataset_name="c4_test",
        dataset_path="./c4_test",
        tokenizer=HuggingFaceTokenizer("./tokenizer"),
        seq_len=seq_len,
        **(
            {
                "dp_rank": torch.distributed.get_rank(),
                "dp_world_size": torch.distributed.get_world_size(),
            }
        ),
        infinite=True,
    )
    return StatefulDataLoader(dataset, batch_size=batch_size)


def trace_handler(prof: Any, traces_dir: pathlib.Path):
    prof.export_chrome_trace(
        str(traces_dir / f"rank{torch.distributed.get_rank()}.json")
    )


def train(
    num_steps: int = 5,
    num_gradient_accumulation_steps: int = 1,
    num_steps_to_profile: int | None = 3,
    lr: float = 1e-3,
    seq_len: int = 256,
    model: Literal["debug", "1b"] = "debug",
    traces_dir: pathlib.Path = pathlib.Path("traces"),
    snapshots_dir: pathlib.Path = pathlib.Path("snapshots"),
    fsdp: Literal["fsdp2", "effdl"] | None = None,
    param_dtype: Literal["bfloat16", "float32"] | None = None,
    reduce_dtype: Literal["bfloat16", "float32"] | None = None,
    reshard_after_forward: bool = True,
    reshard_after_backward_before_last_backward: bool = True,
    reduce_grads_before_last_backward: bool = True,
) -> tuple[list[float], list[float]]:
    if fsdp is not None:
        dp_mesh = init_device_mesh(
            "cuda", mesh_shape=(torch.distributed.get_world_size(),)
        )
        torch.distributed.tensor._random.manual_seed(42, dp_mesh)
    if num_steps_to_profile is not None:
        torch.cuda.memory._record_memory_history()
    with torch.device("cuda"):
        model = Transformer(
            {
                "debug": TransformerModelArgs(
                    dim=256,
                    n_layers=6,
                    n_heads=16,
                    vocab_size=2048,
                    rope_theta=500000,
                ),
                "1b": TransformerModelArgs(
                    dim=2048,
                    n_layers=16,
                    n_heads=32,
                    n_kv_heads=8,
                    ffn_dim_multiplier=1.5,
                    multiple_of=1024,
                    rope_theta=500000,
                ),
            }[model]
        )
    logger.info(f"model size: {sum(p.numel() for p in model.parameters())} parameters.")
    param_dtype = STR_TO_TORCH_DTYPE[param_dtype] if param_dtype is not None else None
    reduce_dtype = (
        STR_TO_TORCH_DTYPE[reduce_dtype] if reduce_dtype is not None else None
    )
    if fsdp == "fsdp2":
        apply_fsdp2(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward=reshard_after_forward,
        )
    elif fsdp == "effdl":
        apply_effdl_fsdp(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward=reshard_after_forward,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = build_dataloader(seq_len=seq_len)
    data_iterator = iter(dataloader)
    traces_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    if num_steps_to_profile is not None:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=functools.partial(trace_handler, traces_dir=traces_dir),
        )
        profiler.start()
    losses: list[float] = []
    grad_norms: list[float] = []
    for step in range(1, num_steps + 1):
        losses: list[torch.Tensor] = []
        for gas_step in range(num_gradient_accumulation_steps):
            is_last_backward = gas_step == num_gradient_accumulation_steps - 1
            for module in model.modules():
                if isinstance(module, FSDP2Module):
                    if reshard_after_backward_before_last_backward is False:
                        module.set_reshard_after_backward(is_last_backward)
                    if reduce_grads_before_last_backward is False:
                        module.set_requires_gradient_sync(is_last_backward)
                if isinstance(module, EffdlFSDPModule):
                    if reshard_after_backward_before_last_backward is False:
                        module.reshard_after_backward = is_last_backward
                    if reduce_grads_before_last_backward is False:
                        module.reduce_grads = is_last_backward
            batch = next(data_iterator)
            inputs, labels = batch
            pred = model(inputs["input"].to("cuda"))
            loss = cross_entropy_loss(pred, labels.to("cuda")) / labels.numel()
            del pred
            loss.backward()
            loss = loss.detach()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        if fsdp is not None:
            loss = all_reduce(
                loss,
                reduceOp="avg",
                group=torch.distributed.distributed_c10d._get_default_group(),
            )
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()])
        if fsdp is not None:
            grad_norm = grad_norm.full_tensor()
        optimizer.step()
        optimizer.zero_grad()
        logger.info(
            f"step: {step:2}  loss: {loss.item():7.4f}  grad_norm: {grad_norm.item():7.4f}"
        )
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        if num_steps_to_profile is not None and step == num_steps_to_profile:
            profiler.stop()
            with open(
                snapshots_dir / f"rank{torch.distributed.get_rank()}.pickle",
                "wb",
            ) as output:
                pickle.dump(torch.cuda.memory._snapshot(), output, protocol=4)
    return losses, grad_norms


if __name__ == "__main__":
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        torch.distributed.init_process_group(backend="nccl")
    tyro.cli(train)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
