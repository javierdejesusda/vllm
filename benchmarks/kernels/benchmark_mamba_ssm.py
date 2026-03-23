# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline tuning and benchmarking script for the Mamba selective_state_update
kernel.

Follows the same pattern as benchmark_moe.py: uses Ray workers to benchmark
different (BLOCK_SIZE_M, num_warps) configurations and writes optimal configs
to JSON files that are loaded at runtime by get_mamba_ssm_configs().

Usage:
    # Benchmark with current config
    python benchmarks/kernels/benchmark_mamba_ssm.py \\
        --model state-spaces/mamba2-2.7b

    # Tune and save optimal configs
    python benchmarks/kernels/benchmark_mamba_ssm.py \\
        --model state-spaces/mamba2-2.7b --tune \\
        --save-dir vllm/model_executor/layers/mamba/ops/configs/
"""

import gc
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, TypedDict

import ray
import torch
from ray.experimental.tqdm_ray import tqdm

import vllm.envs as envs
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    get_mamba_config_file_name,
    get_mamba_ssm_configs,
    selective_state_update,
)
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed


class MambaSSMConfig(TypedDict):
    BLOCK_SIZE_M: int
    num_warps: int


def get_search_space() -> list[MambaSSMConfig]:
    """Generate all (BLOCK_SIZE_M, num_warps) combinations to benchmark."""
    configs = []
    for block_size_m in [4, 8, 16, 32, 64]:
        for num_warps in [1, 2, 4, 8]:
            configs.append(
                MambaSSMConfig(BLOCK_SIZE_M=block_size_m, num_warps=num_warps)
            )
    return configs


def benchmark_config(
    config: MambaSSMConfig,
    batch_size: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    dtype: torch.dtype,
    num_iters: int = 100,
) -> float:
    """Benchmark a single (BLOCK_SIZE_M, num_warps) configuration.

    Creates fresh tensors for every call to avoid the state-overwrite issue
    that prevents using triton.autotune. Uses a temporary config file to
    inject the specific (BLOCK_SIZE_M, num_warps) into the kernel.
    """
    device = torch.device("cuda")

    state = torch.randn(
        batch_size, nheads, head_dim, dstate, dtype=dtype, device=device
    )
    x = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
    dt = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
    A = torch.randn(nheads, head_dim, dstate, dtype=dtype, device=device)
    B = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    D = torch.randn(nheads, head_dim, dtype=dtype, device=device)
    z = torch.randn(batch_size, nheads, head_dim, dtype=dtype, device=device)
    dt_bias = torch.randn(nheads, head_dim, dtype=dtype, device=device)
    out = torch.empty_like(x)

    # Write a temporary config file so selective_state_update uses our
    # specific (BLOCK_SIZE_M, num_warps) for this benchmark run.
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = get_mamba_config_file_name(head_dim, dstate)
        config_path = os.path.join(tmpdir, fname)
        config_data = {str(batch_size): dict(config)}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Patch the config folder and clear the cache so the new config
        # is picked up on the next call.
        saved_folder = envs.VLLM_TUNED_CONFIG_FOLDER
        envs.VLLM_TUNED_CONFIG_FOLDER = tmpdir
        get_mamba_ssm_configs.cache_clear()

        try:
            # Warmup
            for _ in range(5):
                state_copy = state.clone()
                selective_state_update(
                    state_copy,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    out=out,
                )
            torch.cuda.synchronize()

            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iters):
                state_copy = state.clone()
                selective_state_update(
                    state_copy,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    out=out,
                )
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            return elapsed_ms / num_iters * 1000  # microseconds
        finally:
            envs.VLLM_TUNED_CONFIG_FOLDER = saved_folder
            get_mamba_ssm_configs.cache_clear()


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        set_random_seed(seed)
        self.seed = seed
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        batch_size: int,
        nheads: int,
        head_dim: int,
        dstate: int,
        ngroups: int,
        dtype: torch.dtype,
    ) -> tuple[dict[str, int], float]:
        set_random_seed(self.seed)
        op_config = get_mamba_ssm_configs(head_dim, dstate)
        if op_config is None:
            config = MambaSSMConfig(BLOCK_SIZE_M=4, num_warps=8)
        else:
            closest = min(op_config.keys(), key=lambda x: abs(x - batch_size))
            config = op_config[closest]
        kernel_time = benchmark_config(
            config, batch_size, nheads, head_dim, dstate, ngroups, dtype
        )
        return config, kernel_time

    def tune(
        self,
        batch_size: int,
        nheads: int,
        head_dim: int,
        dstate: int,
        ngroups: int,
        dtype: torch.dtype,
        search_space: list[MambaSSMConfig],
    ) -> MambaSSMConfig:
        best_config = None
        best_time = float("inf")
        for config in tqdm(search_space):
            try:
                kernel_time = benchmark_config(
                    config,
                    batch_size,
                    nheads,
                    head_dim,
                    dstate,
                    ngroups,
                    dtype,
                    num_iters=20,
                )
            except Exception:
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

        gc.collect()
        torch.cuda.empty_cache()

        now = datetime.now()
        print(f"[{now.ctime()}] Completed tuning for batch_size={batch_size}")
        assert best_config is not None
        return best_config


def save_configs(
    configs: dict[str, MambaSSMConfig],
    head_dim: int,
    dstate: int,
    save_dir: str,
) -> None:
    filename = get_mamba_config_file_name(head_dim, dstate)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    print(f"Writing best config to {filepath}...")
    with open(filepath, "w") as f:
        json.dump(
            {"triton_version": triton.__version__, **configs},
            f,
            indent=4,
        )
        f.write("\n")


def get_mamba_model_params(config) -> tuple[int, int, int, int]:
    """Extract (nheads, head_dim, dstate, ngroups) from HuggingFace config."""
    architectures = getattr(config, "architectures", None) or [
        type(config).__name__
    ]
    architecture = architectures[0]

    if architecture in ("Mamba2ForCausalLM",):
        nheads = config.num_heads
        head_dim = config.head_dim
        dstate = config.state_size
        ngroups = config.n_groups
    elif architecture in ("JambaForCausalLM",):
        nheads = config.mamba_num_heads
        head_dim = config.mamba_head_dim
        dstate = config.mamba_d_state
        ngroups = config.mamba_n_groups
    elif hasattr(config, "num_heads") and hasattr(config, "state_size"):
        nheads = config.num_heads
        head_dim = getattr(config, "head_dim", 64)
        dstate = config.state_size
        ngroups = getattr(config, "n_groups", 1)
    else:
        raise ValueError(
            f"Cannot extract Mamba SSM parameters from architecture "
            f"{architecture}. Please specify --head-dim, --dstate, "
            f"--nheads, and --ngroups manually."
        )

    return nheads, head_dim, dstate, ngroups


def main(args):
    print(args)

    if (
        args.head_dim is not None
        and args.dstate is not None
        and args.nheads is not None
        and args.ngroups is not None
    ):
        nheads = args.nheads
        head_dim = args.head_dim
        dstate = args.dstate
        ngroups = args.ngroups
    else:
        config = get_config(
            model=args.model, trust_remote_code=args.trust_remote_code
        )
        nheads, head_dim, dstate, ngroups = get_mamba_model_params(config)

    print(
        f"Model params: nheads={nheads}, head_dim={head_dim}, "
        f"dstate={dstate}, ngroups={ngroups}"
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.batch_size is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        batch_sizes = args.batch_size

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.tune:
        search_space = get_search_space()
        print(f"Start tuning over {len(search_space)} configurations...")
        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    nheads,
                    head_dim,
                    dstate,
                    ngroups,
                    dtype,
                    search_space,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            str(M): dict(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(best_configs, head_dim, dstate, args.save_dir)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        outputs = _distribute(
            "benchmark",
            [
                (batch_size, nheads, head_dim, dstate, ngroups, dtype)
                for batch_size in batch_sizes
            ],
        )
        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark and tune Mamba selective_state_update kernel"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba2-2.7b",
        help="HuggingFace model name to extract SSM parameters from",
    )
    parser.add_argument(
        "--nheads", type=int, default=None, help="Number of SSM heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=None, help="SSM head dimension"
    )
    parser.add_argument(
        "--dstate", type=int, default=None, help="SSM state dimension"
    )
    parser.add_argument(
        "--ngroups", type=int, default=None, help="Number of SSM groups"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="Directory to save tuned config JSON files",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, nargs="+", required=False)
    parser.add_argument(
        "--tune", action="store_true", help="Run tuning instead of benchmarking"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    main(args)
