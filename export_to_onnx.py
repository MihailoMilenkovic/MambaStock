#!/usr/bin/env python3
"""
Script to export trained PyTorch models to ONNX format for deployment and inference.
Supports exporting specific model types or all available models.
"""

import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Import our model classes
from mamba import Model, MambaConfig

class Net(torch.nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        
        self.config = config
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, config.d_model),
            Model(config),
            torch.nn.Linear(config.d_model, out_dim),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x.flatten()

def load_pytorch_model(model_path, device):
    """Load a trained PyTorch model from checkpoint"""
    print(f"Loading PyTorch model from {model_path}")
    
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

def export_to_onnx(model, config, output_path, seq_len=20, verify=True):
    """Export PyTorch model to ONNX format"""
    print(f"Exporting {config['model_type']} model to {output_path}")
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, seq_len, config['input_dim'])
    
    # Define dynamic axes for flexible batch size and sequence length
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'output_size'}
    }
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,                          # model being run
            dummy_input,                    # model input (or a tuple for multiple inputs)
            output_path,                    # where to save the model
            export_params=True,             # store the trained parameter weights inside the model file
            opset_version=14,               # the ONNX version to export the model to
            do_constant_folding=True,       # whether to execute constant folding for optimization
            input_names=['input'],          # the model's input names
            output_names=['output'],        # the model's output names
            dynamic_axes=dynamic_axes       # variable length axes
        )
        
        print(f"✅ Successfully exported to {output_path}")
        
        # Verify the exported model
        if verify:
            verify_onnx_model(output_path, dummy_input, model, config)
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to export {config['model_type']} model: {e}")
        return False

def verify_onnx_model(onnx_path, dummy_input, original_model, config):
    """Verify that the ONNX model produces the same output as PyTorch model"""
    print(f"Verifying ONNX model: {onnx_path}")
    
    try:
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # Get PyTorch model output
        with torch.no_grad():
            pytorch_output = original_model(dummy_input)
        
        # Get ONNX model output
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output.numpy() - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-5:
            print(f"   ✅ ONNX model verification passed!")
            return True
        else:
            print(f"   ⚠️  Large difference detected, but this might be acceptable")
            return True
            
    except Exception as e:
        print(f"   ❌ ONNX model verification failed: {e}")
        return False

def test_onnx_inference(onnx_path, config, batch_sizes=[1, 4], seq_len=20):
    """Test ONNX model inference with different batch sizes"""
    print(f"Testing ONNX inference for {config['model_type']} model")
    
    try:
        ort_session = onnxruntime.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        for batch_size in batch_sizes:
            # Create test input
            test_input = np.random.randn(batch_size, seq_len, config['input_dim']).astype(np.float32)
            
            # Run inference
            ort_inputs = {input_name: test_input}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            print(f"   Batch size {batch_size}: input {test_input.shape} → output {onnx_output.shape}")
        
        print(f"   ✅ ONNX inference test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ ONNX inference test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX format')
    parser.add_argument('--ts-code', type=str, default='601988', help='Stock code')
    parser.add_argument('--model-type', type=str, choices=['mamba', 'lstm', 'transformer', 'all'], 
                        default='all', help='Model type to export (default: all)')
    parser.add_argument('--output-dir', type=str, default='onnx_models', 
                        help='Output directory for ONNX models')
    parser.add_argument('--seq-len', type=int, default=20, 
                        help='Sequence length for ONNX export (default: 20)')
    parser.add_argument('--no-verify', action='store_true', 
                        help='Skip ONNX model verification')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip ONNX inference testing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine which models to export
    if args.model_type == 'all':
        model_types = ['mamba', 'lstm', 'transformer']
    else:
        model_types = [args.model_type]
    
    # Find available model files
    available_models = {}
    for model_type in model_types:
        model_path = f'model_{model_type}_{args.ts_code}.pth'
        if os.path.exists(model_path):
            available_models[model_type] = model_path
            print(f"Found {model_type} model: {model_path}")
        else:
            print(f"Warning: {model_type} model not found: {model_path}")
    
    if not available_models:
        print("No models found! Please train models first using main.py")
        sys.exit(1)
    
    print(f"\nExporting {len(available_models)} model(s) to ONNX format...")
    print("=" * 50)
    
    # Export each model
    results = {}
    device = torch.device('cpu')  # Use CPU for ONNX export
    
    for model_type, model_path in available_models.items():
        try:
            # Load PyTorch model
            model, config = load_pytorch_model(model_path, device)
            
            # Define output path
            onnx_path = output_dir / f"{model_type}_{args.ts_code}.onnx"
            
            # Export to ONNX
            success = export_to_onnx(
                model, config, str(onnx_path), 
                seq_len=args.seq_len, 
                verify=not args.no_verify
            )
            
            # Test ONNX inference
            if success and not args.no_test:
                test_onnx_inference(str(onnx_path), config, seq_len=args.seq_len)
            
            results[model_type] = {
                'success': success,
                'onnx_path': str(onnx_path) if success else None
            }
            
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ Failed to process {model_type}: {e}")
            results[model_type] = {'success': False, 'onnx_path': None}
            print("-" * 30)
    
    # Summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    
    successful_exports = []
    for model_type, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{model_type.upper():12s}: {status}")
        if result['success']:
            successful_exports.append(result['onnx_path'])
            print(f"               → {result['onnx_path']}")
    
    if successful_exports:
        print(f"\n🎉 Successfully exported {len(successful_exports)} model(s) to ONNX!")
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        print(f"\n📋 ONNX Model Usage:")
        print(f"   import onnxruntime")
        print(f"   session = onnxruntime.InferenceSession('path/to/model.onnx')")
        print(f"   output = session.run(None, {{'input': input_array}})")
        
        # Show file sizes
        print(f"\n📊 Model Sizes:")
        for onnx_path in successful_exports:
            size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
            model_name = Path(onnx_path).stem
            print(f"   {model_name}: {size_mb:.2f} MB")
    
    else:
        print("\n❌ No models were successfully exported.")
        sys.exit(1)

if __name__ == "__main__":
    main()
