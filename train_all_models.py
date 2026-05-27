#!/usr/bin/env python3
"""
Script to train all model types (mamba, lstm, transformer) with the same configuration
for fair comparison in benchmarks.
"""

import subprocess
import sys
import argparse

def run_training(model_type, args):
    """Run training for a specific model type"""
    cmd = [
        sys.executable, 'main.py',
        '--model_type', model_type,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--hidden', str(args.hidden),
        '--layer', str(args.layer),
        '--ts-code', args.ts_code,
        '--n-test', str(args.n_test)
    ]
    
    if args.use_cuda:
        cmd.append('--use-cuda')
    
    print(f"\nTraining {model_type.upper()} model...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ {model_type.upper()} training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_type.upper()} training failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train all model types for benchmarking')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden dimension')
    parser.add_argument('--layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--ts-code', type=str, default='601988', help='Stock code')
    parser.add_argument('--n-test', type=int, default=300, help='Size of test set')
    
    args = parser.parse_args()
    
    model_types = ['mamba', 'lstm', 'transformer']
    
    print("Training all model types with the following configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dimension: {args.hidden}")
    print(f"  Number of layers: {args.layer}")
    print(f"  Stock code: {args.ts_code}")
    print(f"  Test set size: {args.n_test}")
    print(f"  Use CUDA: {args.use_cuda}")
    print("=" * 50)
    
    results = {}
    
    for model_type in model_types:
        success = run_training(model_type, args)
        results[model_type] = success
    
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    
    for model_type, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_type.upper():12s}: {status}")
    
    successful_models = [mt for mt, success in results.items() if success]
    
    if successful_models:
        print(f"\n{len(successful_models)} model(s) trained successfully!")
        print("You can now run benchmarks with:")
        print(f"python inference_benchmark.py --ts-code {args.ts_code}")
        if args.use_cuda:
            print("                              --use-cuda")
    else:
        print("\nNo models were trained successfully. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
