#!/usr/bin/env python3
"""
Demonstration script showing how to use the exported ONNX models for inference.
"""

import numpy as np
import onnxruntime
import argparse
import os
import time
from pathlib import Path

def load_onnx_model(onnx_path):
    """Load an ONNX model using ONNX Runtime"""
    print(f"Loading ONNX model: {onnx_path}")
    
    try:
        session = onnxruntime.InferenceSession(onnx_path)
        
        # Print model information
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"  Input: {input_info.name} {input_info.shape} ({input_info.type})")
        print(f"  Output: {output_info.name} {output_info.shape} ({output_info.type})")
        
        return session
        
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return None

def run_inference(session, input_data):
    """Run inference on ONNX model"""
    input_name = session.get_inputs()[0].name
    
    # Run inference
    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: input_data})
    end_time = time.perf_counter()
    
    inference_time = (end_time - start_time) * 1000  # Convert to ms
    return outputs[0], inference_time

def main():
    parser = argparse.ArgumentParser(description='Demonstrate ONNX model inference')
    parser.add_argument('--model-dir', type=str, default='onnx_models', 
                        help='Directory containing ONNX models')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size for inference')
    parser.add_argument('--seq-len', type=int, default=20, 
                        help='Sequence length')
    parser.add_argument('--num-runs', type=int, default=10, 
                        help='Number of inference runs for timing')
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist!")
        return
    
    # Find all ONNX models
    onnx_files = list(model_dir.glob('*.onnx'))
    
    if not onnx_files:
        print(f"No ONNX models found in {model_dir}")
        return
    
    print(f"Found {len(onnx_files)} ONNX model(s)")
    print("=" * 50)
    
    # Create sample input data
    # Note: Using 15 features based on the stock data
    input_data = np.random.randn(args.batch_size, args.seq_len, 15).astype(np.float32)
    print(f"Input shape: {input_data.shape}")
    print("=" * 50)
    
    results = {}
    
    # Test each model
    for onnx_path in sorted(onnx_files):
        model_name = onnx_path.stem
        print(f"\nTesting {model_name.upper()} model:")
        print("-" * 30)
        
        # Load model
        session = load_onnx_model(str(onnx_path))
        if session is None:
            continue
        
        try:
            # Run inference multiple times for timing
            times = []
            outputs = None
            
            for i in range(args.num_runs):
                output, inference_time = run_inference(session, input_data)
                times.append(inference_time)
                if i == 0:  # Keep first output for display
                    outputs = output
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[model_name] = {
                'mean_time': mean_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'output_shape': outputs.shape,
                'output_sample': outputs[:5] if len(outputs) > 5 else outputs
            }
            
            print(f"  Output shape: {outputs.shape}")
            print(f"  Sample output: {outputs[:5]}")
            print(f"  Inference time: {mean_time:.2f}ms ± {std_time:.2f}ms")
            print(f"    (min: {min_time:.2f}ms, max: {max_time:.2f}ms)")
            
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
    
    # Comparison summary
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        
        # Sort by inference time
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mean_time'])
        
        print(f"{'Model':<12} {'Time (ms)':<12} {'Speedup':<8}")
        print("-" * 35)
        
        fastest_time = sorted_models[0][1]['mean_time']
        
        for model_name, stats in sorted_models:
            speedup = stats['mean_time'] / fastest_time
            print(f"{model_name:<12} {stats['mean_time']:<8.2f}     {speedup:<6.2f}x")
        
        print(f"\n🚀 {sorted_models[0][0].upper()} is the fastest model!")
        
        # Model sizes
        print(f"\n📊 Model Sizes:")
        for onnx_path in sorted(onnx_files):
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            model_name = onnx_path.stem
            print(f"  {model_name}: {size_mb:.2f} MB")
    
    print(f"\n✅ ONNX inference demonstration completed!")

if __name__ == "__main__":
    main()
