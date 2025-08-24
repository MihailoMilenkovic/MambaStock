import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import argparse
import os
from mamba import Model, MambaConfig

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

def benchmark_model(model, batch_sizes, input_dim, device, num_runs=100):
    """Benchmark model inference for different batch sizes"""
    latencies = {}
    
    # Use a small sequence length for benchmarking to avoid timeout
    seq_len = 20  # This simulates a reasonable sequence length
    
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--ts-code', type=str, default='601988', help='Stock code')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16, 32], 
                        help='Batch sizes to benchmark')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of runs for each benchmark')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_types = ['mamba', 'lstm', 'transformer']
    model_paths = {
        model_type: f'model_{model_type}_{args.ts_code}.pth' 
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
    
    for model_type, model_path in available_models.items():
        print(f"\nBenchmarking {model_type.upper()} model...")
        
        try:
            model, config = load_model(model_path, device)
            latencies = benchmark_model(model, args.batch_sizes, config['input_dim'], device, args.num_runs)
            results[model_type] = latencies
            
            # Print results
            print(f"\n{model_type.upper()} Results:")
            for batch_size, stats in latencies.items():
                print(f"  Batch {batch_size:2d}: {stats['mean']:6.2f}ms ± {stats['std']:5.2f}ms "
                      f"(min: {stats['min']:5.2f}ms, max: {stats['max']:6.2f}ms)")
                
        except Exception as e:
            print(f"Error benchmarking {model_type}: {e}")
            continue
    
    # Create comparison plots
    if len(results) > 1:
        print("\nCreating comparison plots...")
        
        # Plot 1: Mean latency vs batch size
        plt.figure(figsize=(12, 8))
        
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
        
        # Plot 2: Throughput (samples/sec)
        plt.subplot(2, 2, 2)
        for model_type, latencies in results.items():
            batch_sizes = list(latencies.keys())
            throughputs = [bs / (latencies[bs]['mean'] / 1000) for bs in batch_sizes]  # samples/sec
            
            plt.plot(batch_sizes, throughputs, marker='s', label=model_type.upper())
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/sec)')
        plt.title('Inference Throughput vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Latency per sample
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
        
        # Plot 4: Model comparison at specific batch size
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
        
        # Add value labels on bars
        for bar, mean_lat in zip(bars, mean_latencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_latencies[mean_latencies.index(mean_lat)],
                     f'{mean_lat:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'inference_benchmark_{args.ts_code}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Benchmark plots saved as inference_benchmark_{args.ts_code}.png")
    
    # Save benchmark results
    import json
    results_file = f'benchmark_results_{args.ts_code}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'device': str(device),
            'results': results
        }, f, indent=2)
    
    print(f"\nBenchmark results saved to {results_file}")

if __name__ == "__main__":
    main()
