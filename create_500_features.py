#!/usr/bin/env python3
"""
Create proper 500-image gallery features from original dataset
IMPORTANT: This preserves the original file and creates a new 500-image version
"""
import h5py
import numpy as np
from pathlib import Path

def create_500_gallery_features():
    """Create 500-image gallery features from original dataset"""
    
    data_path = Path("data/roxford5k")
    
    # Source: original full dataset (DO NOT MODIFY)
    original_file = data_path / "dinov2_gallery_local.hdf5"
    
    # Destination: new 500-image file
    new_500_file = data_path / "dinov2_gallery_local_500_new.hdf5"
    
    print(f"Creating 500-image features from {original_file}")
    print("="*60)
    
    if not original_file.exists():
        print(f"❌ Original file not found: {original_file}")
        return
    
    with h5py.File(original_file, 'r') as original_f:
        original_data = original_f['features'][:]
        print(f"Original data shape: {original_data.shape}")
        print(f"Original data type: {original_data.dtype}")
        
        # Check if it's structured array (with metadata and descriptor fields)
        if len(original_data.dtype.names or []) > 0:
            print("✅ Found structured array format")
            
            # Extract first 500 images
            data_500 = original_data[:500]
            print(f"Extracted 500 images: {data_500.shape}")
            
            # Create new file with same structure
            with h5py.File(new_500_file, 'w') as new_f:
                new_f.create_dataset('features', data=data_500)
                print(f"✅ Created {new_500_file}")
        
        else:
            print("✅ Found simple array format")
            # Handle simple array format (num_images, topk, feature_dim)
            data_500 = original_data[:500]
            print(f"Extracted 500 images: {data_500.shape}")
            
            with h5py.File(new_500_file, 'w') as new_f:
                new_f.create_dataset('features', data=data_500)
                print(f"✅ Created {new_500_file}")
    
    # Replace the old 500 file with the new one
    old_500_file = data_path / "dinov2_gallery_local_500.hdf5"
    if old_500_file.exists():
        backup_file = data_path / "dinov2_gallery_local_500_backup.hdf5"
        old_500_file.rename(backup_file)
        print(f"📦 Backed up old file to {backup_file}")
    
    new_500_file.rename(old_500_file)
    print(f"✅ Renamed to {old_500_file}")
    
    # Verify the new file
    print(f"\n🔍 Verifying new 500-image file...")
    with h5py.File(old_500_file, 'r') as f:
        data = f['features'][:]
        print(f"✅ Final shape: {data.shape}")
        print(f"✅ Final dtype: {data.dtype}")
        
        actual_size = old_500_file.stat().st_size
        print(f"✅ File size: {actual_size / 1024 / 1024:.1f} MB")
        
        # Count actual images if structured
        if len(data.dtype.names or []) > 0:
            print(f"✅ Contains {len(data)} images with structured format")
        else:
            print(f"✅ Contains {data.shape[0]} images")

if __name__ == "__main__":
    create_500_gallery_features()
