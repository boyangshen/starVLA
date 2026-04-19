#!/usr/bin/env python3
"""
Profile FLOPs and Peak GPU Memory for StarVLA model.

Usage:
    python profile_flops_memory.py --config_yaml <config> --checkpoint <checkpoint> [--num_samples 10]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_thop_installed():
    """Check and install thop if not available."""
    try:
        import thop
        return True
    except ImportError:
        print("[INFO] thop not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"], stdout=subprocess.DEVNULL)
            print("[INFO] thop installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("[WARN] Failed to install thop, will use manual estimation")
            return False


ensure_thop_installed()

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FLOPs and GPU Memory for StarVLA")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to YAML config (optional if checkpoint is provided)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (optional)",
    )
    parser.add_argument(
        "--dataset_yaml",
        type=str,
        default=None,
        help="Path to dataset-only YAML config (for data loading, model from checkpoint)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to profile",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/profiling",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="profiling_results.json",
        help="Output filename for profiling results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for profiling",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--profile_module",
        type=str,
        default=None,
        help="Module name to profile (e.g., 'qwen_vl_interface'). If None, profile full model.",
    )
    args, unknown = parser.parse_known_args()
    return args


def setup_model_and_data(args):
    """Build model and dataloader."""
    import os
    import torch.distributed as dist
    from starVLA.model.framework.base_framework import baseframework

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "24500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if args.checkpoint:
        print(f"[INFO] Loading model from checkpoint: {args.checkpoint}")
        import time
        t0 = time.time()
        model = baseframework.from_pretrained(args.checkpoint)
        print(f"[INFO] Model loaded in {time.time()-t0:.1f}s")
        model.eval()

        if args.config_yaml:
            cfg = OmegaConf.load(args.config_yaml)
            cfg = wrap_config(cfg)
        else:
            cfg = model.config

        dataset_py = cfg.datasets.vla_data.dataset_py if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'vla_data') else "lerobot_datasets"
    elif args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
        cfg = wrap_config(cfg)
        import time
        t0 = time.time()
        model = build_framework(cfg=cfg)
        print(f"[INFO] Model built in {time.time()-t0:.1f}s")
        model.eval()
        dataset_py = cfg.datasets.vla_data.dataset_py if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'vla_data') else "lerobot_datasets"
    else:
        raise ValueError("Must provide either --checkpoint or --config_yaml")

    print(f"[INFO] Building dataloader...")
    import time
    t0 = time.time()
    if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'vla_data'):
        original_batch_size = cfg.datasets.vla_data.per_device_batch_size
        cfg.datasets.vla_data.per_device_batch_size = args.batch_size
        print(f"[INFO] Overriding batch_size: {original_batch_size} -> {args.batch_size}")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=dataset_py)
    print(f"[INFO] Dataloader built in {time.time()-t0:.1f}s")

    return model, vla_train_dataloader


def get_sample_batch(dataloader, num_samples=10, random_seed=42):
    """Get random sample batches using SubsetRandomSampler (fast)."""
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    dataset = dataloader.dataset
    batch_size = dataloader.batch_size

    total_samples = len(dataset)
    num_to_sample = min(num_samples * batch_size, total_samples)

    indices = np.random.choice(total_samples, size=num_to_sample, replace=False).tolist()

    sampler = SubsetRandomSampler(indices)
    shuffled_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=0,
        persistent_workers=False,
    )

    samples = []
    for batch in shuffled_loader:
        samples.append(batch)
        if len(samples) >= num_samples:
            break

    return samples


def measure_peak_memory(model, batch, device):
    """Measure peak GPU memory usage during forward pass."""
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    initial_memory = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        _ = model(batch)

    torch.cuda.synchronize(device)

    peak_memory = torch.cuda.max_memory_allocated(device)
    return peak_memory / (1024 ** 3)


class ModuleProfiler:
    """Profile specific module's forward time using hooks."""
    def __init__(self, model, module_name):
        self.module_name = module_name
        self.module = self._find_module(model, module_name)
        self.times = []
        self.handles = []

    def _find_module(self, model, name):
        for n, m in model.named_modules():
            if n == name:
                return m
        return None

    def _pre_hook_fn(self, module, input):
        torch.cuda.synchronize()
        self.times.append(("start", time.time()))

    def _hook_fn(self, module, input, output):
        torch.cuda.synchronize()
        self.times.append(("end", time.time()))

    def start(self):
        if self.module is None:
            print(f"[WARN] Module '{self.module_name}' not found in model")
            return
        self.handles.append(self.module.register_forward_pre_hook(self._pre_hook_fn))
        self.handles.append(self.module.register_forward_hook(self._hook_fn))
        print(f"[INFO] Registered hooks on module: {self.module_name}")

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_elapsed_times(self):
        if len(self.times) < 2:
            return []
        starts = [t for _, t in self.times if _ == "start"]
        ends = [t for _, t in self.times if _ == "end"]
        if len(starts) != len(ends):
            return []
        return [(ends[i] - starts[i]) * 1000 for i in range(len(starts))]

    def get_avg_time_ms(self):
        times = self.get_elapsed_times()
        return sum(times) / len(times) if times else 0


def compute_flops_thop(model, batch, device):
    """Compute FLOPs using thop library."""
    try:
        from thop import profile

        model.eval()

        if isinstance(batch, dict):
            inputs = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                elif isinstance(value, (list, dict)):
                    inputs[key] = value
            flops, params = profile(model, inputs=(inputs,), verbose=False)
        elif isinstance(batch, (list, tuple)):
            inputs = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    inputs.append(value.to(device))
                else:
                    inputs.append(value)
            flops, params = profile(model, inputs=(inputs,), verbose=False)
        else:
            flops, params = profile(model, inputs=(batch,), verbose=False)

        return flops, params
    except Exception as e:
        print(f"[WARN] thop profiling failed: {e}")
        return None, None


def compute_flops_manual(model, batch, device):
    """Estimate FLOPs manually based on model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch_size = 1
    if "image" in batch:
        if isinstance(batch["image"], torch.Tensor):
            batch_size = batch["image"].shape[0]
        elif isinstance(batch["image"], list):
            batch_size = len(batch["image"])

    hidden_size = 2048
    num_layers = 16
    sequence_length = 1024
    vocab_size = 151936

    estimated_flops = 0

    try:
        vlm_flops = 2 * hidden_size * sequence_length * vocab_size * 1
        estimated_flops += vlm_flops
    except:
        pass

    action_flops = 2 * batch_size * num_layers * hidden_size * sequence_length * 3
    estimated_flops += action_flops

    return estimated_flops, total_params, trainable_params


def profile_model(model, dataloader, args, device):
    """Run full profiling."""
    print(f"\n{'='*60}")
    print("StarVLA Model Profiling")
    print(f"{'='*60}\n")

    print(f"[INFO] Sampling {args.num_samples} batches with seed {args.random_seed}...")
    samples = get_sample_batch(dataloader, args.num_samples, args.random_seed)
    print(f"[INFO] Sampled {len(samples)} batches")

    if not samples:
        print("[ERROR] No samples available in dataloader")
        return

    batch = samples[0]

    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(batch, (list, tuple)):
        print(f"  batch: list of length {len(batch)}")
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")

    print(f"\n[INFO] Moving model to {device}...")
    import time
    t0 = time.time()
    model = model.to(device)
    torch.cuda.synchronize(device)
    print(f"[INFO] Moved model in {time.time()-t0:.1f}s")

    model.eval()

    print("[INFO] Measuring peak memory...")
    t0 = time.time()
    peak_memory_gb = measure_peak_memory(model, batch, device)
    print(f"[INFO] Peak Memory: {peak_memory_gb:.2f} GB")

    print("[INFO] Computing FLOPs...")
    t0 = time.time()
    thop_flops, thop_params = compute_flops_thop(model, batch, device)
    if thop_flops is not None:
        print(f"[INFO] FLOPs (thop): {thop_flops / 1e12:.2f} T")
        print(f"[INFO] Params (thop): {thop_params / 1e6:.2f} M")
    else:
        print(f"[WARN] thop failed, using manual estimation")
        manual_flops, total_params, trainable_params = compute_flops_manual(model, batch, device)
        print(f"[INFO] FLOPs (estimated): {manual_flops / 1e12:.2f} T")
        print(f"[INFO] Total Params: {total_params / 1e6:.2f} M")
        print(f"[INFO] Trainable Params: {trainable_params / 1e6:.2f} M")

    print(f"\n[INFO] Profiling {len(samples)} samples for memory & latency stats...")
    memory_usage = []
    inference_times = []

    module_profiler = None
    if args.profile_module:
        module_profiler = ModuleProfiler(model, args.profile_module)
        module_profiler.start()

    for i, sample in enumerate(tqdm(samples, desc="Profiling samples")):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        t_start = time.time()
        with torch.no_grad():
            _ = model(sample)
        torch.cuda.synchronize(device)
        t_end = time.time()

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        infer_time = (t_end - t_start) * 1000
        memory_usage.append(peak_mem)
        inference_times.append(infer_time)

    if module_profiler:
        module_profiler.stop()
        module_times = module_profiler.get_elapsed_times()
        if module_times:
            print(f"\n[INFO] Module '{args.profile_module}' Time Statistics:")
            print(f"  Min Module Time: {min(module_times):.2f} ms")
            print(f"  Max Module Time: {max(module_times):.2f} ms")
            print(f"  Avg Module Time: {sum(module_times) / len(module_times):.2f} ms")

    print(f"\n[INFO] Memory Statistics:")
    print(f"  Min Peak Memory: {min(memory_usage):.2f} GB")
    print(f"  Max Peak Memory: {max(memory_usage):.2f} GB")
    print(f"  Avg Peak Memory: {sum(memory_usage) / len(memory_usage):.2f} GB")

    print(f"\n[INFO] Latency Statistics:")
    print(f"  Min Inference Time: {min(inference_times):.2f} ms")
    print(f"  Max Inference Time: {max(inference_times):.2f} ms")
    print(f"  Avg Inference Time: {sum(inference_times) / len(inference_times):.2f} ms")
    avg_time_ms = sum(inference_times) / len(inference_times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    print(f"  FPS (throughput): {fps:.2f}")

    results = {
        "peak_memory_gb": peak_memory_gb,
        "num_samples_profiled": len(samples),
        "memory_usage_samples": memory_usage,
        "min_peak_memory_gb": min(memory_usage),
        "max_peak_memory_gb": max(memory_usage),
        "avg_peak_memory_gb": sum(memory_usage) / len(memory_usage),
        "inference_times_ms": inference_times,
        "min_inference_time_ms": min(inference_times),
        "max_inference_time_ms": max(inference_times),
        "avg_inference_time_ms": avg_time_ms,
        "fps": fps,
    }

    if module_profiler and module_times:
        results[f"module_{args.profile_module}_times_ms"] = module_times
        results[f"min_module_time_ms"] = min(module_times)
        results[f"max_module_time_ms"] = max(module_times)
        results[f"avg_module_time_ms"] = sum(module_times) / len(module_times)

    if thop_flops is not None:
        results["flops_t"] = thop_flops / 1e12
        results["params_m"] = thop_params / 1e6

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_filename

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Results saved to {output_file}")

    print(f"\n{'='*60}")
    print("Profiling Complete!")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script requires GPU.")
        return

    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"[INFO] Building model and dataloader...")
    model, dataloader = setup_model_and_data(args)

    profile_model(model, dataloader, args, device)


if __name__ == "__main__":
    main()
