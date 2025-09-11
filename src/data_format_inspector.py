#!/usr/bin/env python3
"""
Quick data format inspector to understand your HDF5 and pickle files
Run this first to understand the data structure


DATA FORMAT INSPECTOR
============================================================

==================== dinov2_gallery_local.hdf5 ====================
Keys: ['features']

Dataset: features
  Shape: (4993,)
  Dtype: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]
  Size: 5.06 GB

==================== dinov2_query_local.hdf5 ====================
Keys: ['features']

Dataset: features
  Shape: (70,)
  Dtype: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]
  Size: 0.07 GB

==================== gnd_roxford5k.pkl ====================
Type: <class 'dict'>
Keys: ['gnd', 'imlist', 'qimlist']
Number of queries: 70
First few queries: ['all_souls_000013', 'all_souls_000026', 'oxford_002985']
Number of gallery images: 4993
First few gallery: ['ashmolean_000283', 'oxford_002124', 'radcliffe_camera_000158']
Ground truth entries: 70
First entry keys: ['bbx', 'easy', 'hard', 'junk']
First entry 'bbx' count: 4
First entry 'easy' count: 65
First entry 'hard' count: 38
First entry 'junk' count: 33

==================== nn_dinov2.pkl ====================
Type: <class 'torch.Tensor'>

==================== test_gallery.txt ====================
Total lines: 4993
First 5 lines:
  1: jpg/ashmolean_000283.jpg,1,1024,768
  2: jpg/oxford_002124.jpg,12,1024,683
  3: jpg/radcliffe_camera_000158.jpg,14,740,1024
  4: jpg/oxford_002084.jpg,12,1024,768
  5: jpg/oxford_003332.jpg,12,1024,819
  ...
  4993: jpg/christ_church_000907.jpg,4,768,1024

==================== test_query.txt ====================
Total lines: 70
First 5 lines:
  1: jpg/all_souls_000013.jpg,0,768,1024
  2: jpg/all_souls_000026.jpg,0,819,1024
  3: jpg/oxford_002985.jpg,12,768,1024
  4: jpg/all_souls_000051.jpg,0,1024,768
  5: jpg/oxford_003410.jpg,12,683,1024
  ...
  70: jpg/radcliffe_camera_000031.jpg,14,686,1024

============================================================
SUMMARY
============================================================
Based on inspection, the typical structure should be:
• Gallery features: [N_gallery, patches, 768] in HDF5
• Query features: [N_query, patches, 768] in HDF5
• Ground truth: dict with 'qimlist', 'imlist', 'gnd' keys
• Text files: Simple lists of image names


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
            first_gnd = data['gnd'][0]
            print(f"First entry keys: {list(first_gnd.keys())}")
            for k in first_gnd:
                v = first_gnd[k]
                if isinstance(v, (list, np.ndarray)):
                    print(f"First entry '{k}' count: {len(v)}")
                else:
                    print(f"First entry '{k}': {v}")

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
