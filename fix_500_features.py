#!/usr/bin/env python3
"""
Extract exactly 500 images from the full gallery features
"""
import h5py
import numpy as np
from pathlib import Path

def extract_500_features():
    """Extract first 500 images from full gallery features"""
    
    data_path = Path("data/roxford5k")
    
    # Source: full dataset (copied from original)
    src_file = data_path / "dinov2_gallery_local_500.hdf5"  # This is actually the full dataset
    
    # Destination: actual 500-image file
    dst_file = data_path / "dinov2_gallery_local_500_fixed.hdf5"
    
    print(f"Extracting 500 images from {src_file}")
    print("="*60)
    
    with h5py.File(src_file, 'r') as src_f:
        full_data = src_f['features'][:]
        print(f"Source data shape: {full_data.shape}")
        print(f"Source data type: {full_data.dtype}")
        
        # Extract first 500 images
        data_500 = full_data[:500]
        print(f"Extracted shape: {data_500.shape}")
        
        # Create new file with just 500 images
        with h5py.File(dst_file, 'w') as dst_f:
            dst_f.create_dataset('features', data=data_500)
            print(f"Created {dst_file}")
    
    # Replace the original 500 file
    import shutil
    backup_file = data_path / "dinov2_gallery_local_500_backup.hdf5"
    shutil.move(src_file, backup_file)
    shutil.move(dst_file, src_file)
    
    print(f"✅ Replaced {src_file} with 500-image version")
    print(f"   Original backed up to {backup_file}")
    
    # Verify the new file
    with h5py.File(src_file, 'r') as f:
        data = f['features'][:]
        print(f"✅ Final verification: {data.shape} images")
        
        # Calculate proper file size
        expected_size = 500 * 700 * (5 + 768) * 4  # metadata + descriptors, float32
        actual_size = src_file.stat().st_size
        print(f"   File size: {actual_size / 1024 / 1024:.1f} MB")
        print(f"   Expected: {expected_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    extract_500_features()
