import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import argparse
import os
from mamba import Model, MambaConfig
import onnxruntime as ort
from pathlib import Path

class Net(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(in_dim, config.d_model),
            Model(config),
            nn.Linear(config.d_model, out_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.flatten()

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model config
    model_config = MambaConfig(
        d_model=config['hidden'],
        n_layers=config['layer'],
        model_type=config['model_type']
    )
    
    # Create and load model
    model = Net(model_config, config['input_dim'], 1)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def benchmark_model_pytorch(model, batch_sizes, input_dim, device, num_runs=100, seq_len=20):
    """Benchmark PyTorch model inference for different batch sizes"""
    latencies = {}
    # Use a small sequence length for benchmarking to avoid timeout
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size} (PyTorch)...")
        
        # Generate random input: (batch_size, seq_len, input_dim)  
        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies[batch_size] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    return latencies

def benchmark_model_pytorch_by_seq(model, seq_lens, input_dim, device, num_runs=100, batch_size=1):
    """Benchmark PyTorch model inference for different sequence lengths.

    Uses a fixed batch size and sweeps sequence lengths.
    Returns a dict keyed by sequence length with latency stats.
    """
    latencies = {}
    for sl in seq_lens:
        print(f"Benchmarking sequence length {sl} (PyTorch)...")

        x = torch.randn(batch_size, sl, input_dim, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        times = []
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        latencies[sl] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

    return latencies

def load_onnx_model_ort(model_path):
    """Load ONNX model with ONNX Runtime, prefer GPU if available, and infer input_dim."""
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        used_provider = 'CUDAExecutionProvider'
    else:
        providers = ['CPUExecutionProvider']
        used_provider = 'CPUExecutionProvider'
    session = ort.InferenceSession(str(model_path), providers=providers)
    inp = session.get_inputs()[0]
    shape = inp.shape
    input_dim = shape[-1] if isinstance(shape[-1], int) else None
    return session, input_dim, used_provider

def benchmark_model_onnx(session, batch_sizes, input_dim, num_runs=100, seq_len=20):
    """Benchmark ONNX model inference for different batch sizes on CPU."""
    latencies = {}
    input_name = session.get_inputs()[0].name
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size} (ONNX)...")
        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        # Warmup
        for _ in range(3):
            _ = session.run(None, {input_name: x})
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: x})
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        latencies[batch_size] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }
    return latencies

def benchmark_model_onnx_by_seq(session, seq_lens, input_dim, num_runs=100, batch_size=1):
    """Benchmark ONNX model inference for different sequence lengths.

    Uses a fixed batch size and sweeps sequence lengths.
    Returns a dict keyed by sequence length with latency stats.
    """
    latencies = {}
    input_name = session.get_inputs()[0].name
    for sl in seq_lens:
        print(f"Benchmarking sequence length {sl} (ONNX)...")
        x = np.random.randn(batch_size, sl, input_dim).astype(np.float32)

        # Warmup
        for _ in range(3):
            _ = session.run(None, {input_name: x})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: x})
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        latencies[sl] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }
    return latencies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', help='[Deprecated] CUDA auto-detected; flag ignored')
    parser.add_argument('--ts-code', type=str, default='601988', help='Stock code')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16, 32], 
                        help='Batch sizes to benchmark')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of runs for each benchmark')
    parser.add_argument('--backend', type=str, choices=['pytorch', 'onnx'], default='pytorch',
                        help='Inference backend to use')
    parser.add_argument('--seq-len', type=int, default=20, help='Sequence length for inputs (default 20)')
    parser.add_argument('--vary-sl', action='store_true', help='Vary sequence length instead of batch size')
    parser.add_argument('--onnx-dir', type=str, default='onnx_models', help='Directory with ONNX models')
    
    args = parser.parse_args()
    
    # Auto-select device: use CUDA if available, else CPU (PyTorch). ONNX uses CUDA EP if available.
    device = torch.device('cuda' if args.backend == 'pytorch' and torch.cuda.is_available() else 'cpu')
    if args.backend == 'pytorch':
        print(f"Using device: {device}")
    
    model_types = ['mamba', 'lstm', 'transformer']
    if args.backend == 'pytorch':
        model_paths = {
            model_type: f'model_{model_type}_{args.ts_code}.pth' 
            for model_type in model_types
        }
    else:
        onnx_dir = Path(args.onnx_dir)
        model_paths = {
            model_type: str(onnx_dir / f"{model_type}_{args.ts_code}.onnx")
            for model_type in model_types
        }
    
    # Check which models exist
    available_models = {}
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            available_models[model_type] = path
            print(f"Found {model_type} model: {path}")
        else:
            print(f"Warning: {model_type} model not found: {path}")
    
    if not available_models:
        print("No trained models found! Please train models first using main.py")
        return
    
    # Benchmark each model
    results = {}
    
    onnx_used_provider = None
    for model_type, model_path in available_models.items():
        print(f"\nBenchmarking {model_type.upper()} model...")
        try:
            if args.backend == 'pytorch':
                model, config = load_model(model_path, device)
                if args.vary_sl:
                    seq_lens = [20, 100, 500, 2500, 12500]
                    fixed_batch = args.batch_sizes[0]
                    latencies = benchmark_model_pytorch_by_seq(
                        model,
                        seq_lens,
                        config['input_dim'],
                        device,
                        args.num_runs,
                        batch_size=fixed_batch,
                    )
                else:
                    latencies = benchmark_model_pytorch(
                        model,
                        args.batch_sizes,
                        config['input_dim'],
                        device,
                        args.num_runs,
                        seq_len=args.seq_len,
                    )
            else:
                session, input_dim, provider = load_onnx_model_ort(model_path)
                onnx_used_provider = provider
                if input_dim is None:
                    # Attempt to read input_dim from corresponding checkpoint if available
                    ckpt_path = f'model_{model_type}_{args.ts_code}.pth'
                    if os.path.exists(ckpt_path):
                        try:
                            _, cfg = load_model(ckpt_path, torch.device('cpu'))
                            input_dim = cfg['input_dim']
                        except Exception:
                            pass
                if input_dim is None:
                    input_dim = 15  # Fallback common default in this repo
                if args.vary_sl:
                    seq_lens = [20, 100, 500, 2500, 12500]
                    fixed_batch = args.batch_sizes[0]
                    latencies = benchmark_model_onnx_by_seq(
                        session,
                        seq_lens,
                        input_dim,
                        args.num_runs,
                        batch_size=fixed_batch,
                    )
                else:
                    latencies = benchmark_model_onnx(
                        session,
                        args.batch_sizes,
                        input_dim,
                        args.num_runs,
                        seq_len=args.seq_len,
                    )
            results[model_type] = latencies
            # Print results
            print(f"\n{model_type.upper()} Results:")
            if args.vary_sl:
                for sl, stats in latencies.items():
                    print(
                        f"  SeqLen {sl:5d}: {stats['mean']:6.2f}ms ± {stats['std']:5.2f}ms "
                        f"(min: {stats['min']:5.2f}ms, max: {stats['max']:6.2f}ms)"
                    )
            else:
                for batch_size, stats in latencies.items():
                    print(
                        f"  Batch {batch_size:2d}: {stats['mean']:6.2f}ms ± {stats['std']:5.2f}ms "
                        f"(min: {stats['min']:5.2f}ms, max: {stats['max']:6.2f}ms)"
                    )
        except Exception as e:
            print(f"Error benchmarking {model_type}: {e}")
            continue
    
    # Create comparison plots
    if len(results) > 1:
        print("\nCreating comparison plots...")
        
        # Add a figure-level title with backend and device/provider
        if args.backend == 'pytorch':
            run_label = f"Backend: PyTorch | Device: {device}"
        else:
            dev_label = 'cuda' if (onnx_used_provider and 'CUDA' in onnx_used_provider) else 'cpu'
            run_label = f"Backend: ONNX | Provider: {onnx_used_provider or 'CPUExecutionProvider'} | Device: {dev_label}"

        # Plot 1-4 depending on whether we vary sequence length or batch size
        plt.figure(figsize=(12, 8))

        if args.vary_sl:
            # Plot 1: Mean latency vs sequence length
            plt.subplot(2, 2, 1)
            for model_type, latencies in results.items():
                seq_lens = list(latencies.keys())
                means = [latencies[sl]['mean'] for sl in seq_lens]
                stds = [latencies[sl]['std'] for sl in seq_lens]
                plt.errorbar(seq_lens, means, yerr=stds, marker='o', label=model_type.upper(), capsize=5)

            plt.xlabel('Sequence Length')
            plt.ylabel('Latency (ms)')
            plt.title('Mean Inference Latency vs Sequence Length')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            # Plot 2: Throughput (samples/sec) at fixed batch size
            plt.subplot(2, 2, 2)
            fixed_batch = args.batch_sizes[0]
            for model_type, latencies in results.items():
                seq_lens = list(latencies.keys())
                throughputs = [fixed_batch / (latencies[sl]['mean'] / 1000) for sl in seq_lens]
                plt.plot(seq_lens, throughputs, marker='s', label=model_type.upper())

            plt.xlabel('Sequence Length')
            plt.ylabel('Throughput (samples/sec)')
            plt.title(f'Inference Throughput vs Sequence Length (Batch={fixed_batch})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 3: Latency per sample
            plt.subplot(2, 2, 3)
            for model_type, latencies in results.items():
                seq_lens = list(latencies.keys())
                latency_per_sample = [latencies[sl]['mean'] / args.batch_sizes[0] for sl in seq_lens]
                plt.plot(seq_lens, latency_per_sample, marker='^', label=model_type.upper())

            plt.xlabel('Sequence Length')
            plt.ylabel('Latency per Sample (ms)')
            plt.title(f'Latency per Sample vs Sequence Length (Batch={args.batch_sizes[0]})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 4: Model comparison at specific sequence length
            plt.subplot(2, 2, 4)
            seq_lens_all = list(next(iter(results.values())).keys())
            comparison_seq_len = 500 if 500 in seq_lens_all else seq_lens_all[0]

            model_names = []
            mean_latencies = []
            std_latencies = []
            for model_type, latencies in results.items():
                if comparison_seq_len in latencies:
                    model_names.append(model_type.upper())
                    mean_latencies.append(latencies[comparison_seq_len]['mean'])
                    std_latencies.append(latencies[comparison_seq_len]['std'])

            bars = plt.bar(model_names, mean_latencies, yerr=std_latencies, capsize=5, alpha=0.7)
            plt.ylabel('Latency (ms)')
            plt.title(f'Model Comparison (Seq Len = {comparison_seq_len})')
            plt.grid(True, alpha=0.3, axis='y')

            for bar, mean_lat, std_lat in zip(bars, mean_latencies, std_latencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_lat,
                         f'{mean_lat:.1f}ms', ha='center', va='bottom')
        else:
            # Original batch-size plots
            plt.subplot(2, 2, 1)
            for model_type, latencies in results.items():
                batch_sizes = list(latencies.keys())
                means = [latencies[bs]['mean'] for bs in batch_sizes]
                stds = [latencies[bs]['std'] for bs in batch_sizes]
                plt.errorbar(batch_sizes, means, yerr=stds, marker='o', label=model_type.upper(), capsize=5)

            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            plt.title('Mean Inference Latency vs Batch Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            plt.subplot(2, 2, 2)
            for model_type, latencies in results.items():
                batch_sizes = list(latencies.keys())
                throughputs = [bs / (latencies[bs]['mean'] / 1000) for bs in batch_sizes]
                plt.plot(batch_sizes, throughputs, marker='s', label=model_type.upper())

            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/sec)')
            plt.title('Inference Throughput vs Batch Size')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            for model_type, latencies in results.items():
                batch_sizes = list(latencies.keys())
                latency_per_sample = [latencies[bs]['mean'] / bs for bs in batch_sizes]
                plt.plot(batch_sizes, latency_per_sample, marker='^', label=model_type.upper())

            plt.xlabel('Batch Size')
            plt.ylabel('Latency per Sample (ms)')
            plt.title('Latency per Sample vs Batch Size')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            comparison_batch_size = 16 if 16 in args.batch_sizes else args.batch_sizes[0]

            model_names = []
            mean_latencies = []
            std_latencies = []
            for model_type, latencies in results.items():
                if comparison_batch_size in latencies:
                    model_names.append(model_type.upper())
                    mean_latencies.append(latencies[comparison_batch_size]['mean'])
                    std_latencies.append(latencies[comparison_batch_size]['std'])

            bars = plt.bar(model_names, mean_latencies, yerr=std_latencies, capsize=5, alpha=0.7)
            plt.ylabel('Latency (ms)')
            plt.title(f'Model Comparison (Batch Size = {comparison_batch_size})')
            plt.grid(True, alpha=0.3, axis='y')

            for bar, mean_lat, std_lat in zip(bars, mean_latencies, std_latencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_lat,
                         f'{mean_lat:.1f}ms', ha='center', va='bottom')
        
        plt.suptitle(run_label)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_suffix = 'seq' if args.vary_sl else 'batch'
        plt.savefig(f'inference_benchmark_{args.ts_code}_{out_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Benchmark plots saved as inference_benchmark_{args.ts_code}_{out_suffix}.png")
    
    # Save benchmark results
    import json
    results_file = f'benchmark_results_{args.ts_code}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'backend': args.backend,
            'device': str(device) if args.backend == 'pytorch' else (
                'cuda' if (onnx_used_provider and 'CUDA' in onnx_used_provider) else 'cpu'
            ),
            'onnx_provider': onnx_used_provider,
            'results': results
        }, f, indent=2)
    
    print(f"\nBenchmark results saved to {results_file}")

if __name__ == "__main__":
    main()
