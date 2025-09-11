#!/usr/bin/env python3
"""
Quick data format inspector to understand your HDF5 and pickle files
Run this first to understand the data structure
"""
import h5py
import pickle
import numpy as np
from pathlib import Path

def inspect_hdf5(filepath):
    """Inspect HDF5 file structure"""
    print(f"\n{'='*20} {filepath.name} {'='*20}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        for key in f.keys():
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                print(f"\nDataset: {key}")
                print(f"  Shape: {dataset.shape}")
                print(f"  Dtype: {dataset.dtype}")
                print(f"  Size: {dataset.size * dataset.dtype.itemsize / (1024**3):.2f} GB")
                
                # Sample some values
                if len(dataset.shape) == 3:  # [num_images, patches, features]
                    print(f"  First image shape: {dataset[0].shape}")
                    print(f"  First patch, first 5 features: {dataset[0, 0, :5]}")
                elif len(dataset.shape) == 2:  # [num_images, features]
                    print(f"  First image, first 5 features: {dataset[0, :5]}")

def inspect_pickle(filepath):
    """Inspect pickle file"""
    print(f"\n{'='*20} {filepath.name} {'='*20}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        # Common keys in ROxford5k ground truth
        if 'qimlist' in data:
            print(f"Number of queries: {len(data['qimlist'])}")
            print(f"First few queries: {data['qimlist'][:3]}")
        
        if 'imlist' in data:
            print(f"Number of gallery images: {len(data['imlist'])}")
            print(f"First few gallery: {data['imlist'][:3]}")
            
        if 'gnd' in data:
            print(f"Ground truth entries: {len(data['gnd'])}")
            print(f"First entry keys: {list(data['gnd'][0].keys())}")
            print(f"First entry 'ok' count: {len(data['gnd'][0]['ok'])}")

def inspect_text_file(filepath):
    """Inspect text file"""
    print(f"\n{'='*20} {filepath.name} {'='*20}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print(f"First 5 lines:")
    for i, line in enumerate(lines[:5]):
        print(f"  {i+1}: {line.strip()}")
    
    if len(lines) > 5:
        print(f"  ...")
        print(f"  {len(lines)}: {lines[-1].strip()}")

def main():
    """Inspect all data files"""
    data_root = Path(r"C:\github\ames\ames\data\roxford5k")
    
    print("DATA FORMAT INSPECTOR")
    print("="*60)
    
    # Files to inspect
    files_to_check = [
        # HDF5 files
        ("dinov2_gallery_local.hdf5", inspect_hdf5),
        ("dinov2_query_local.hdf5", inspect_hdf5),
        
        # Pickle files
        ("gnd_roxford5k.pkl", inspect_pickle),
        ("nn_dinov2.pkl", inspect_pickle),
        
        # Text files
        ("test_gallery.txt", inspect_text_file),
        ("test_query.txt", inspect_text_file),
    ]
    
    for filename, inspector_func in files_to_check:
        filepath = data_root / filename
        if filepath.exists():
            try:
                inspector_func(filepath)
            except Exception as e:
                print(f"Error inspecting {filename}: {e}")
        else:
            print(f"\n⚠️ File not found: {filename}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print("Based on inspection, the typical structure should be:")
    print("• Gallery features: [N_gallery, patches, 768] in HDF5")
    print("• Query features: [N_query, patches, 768] in HDF5") 
    print("• Ground truth: dict with 'qimlist', 'imlist', 'gnd' keys")
    print("• Text files: Simple lists of image names")

if __name__ == "__main__":
    main()
