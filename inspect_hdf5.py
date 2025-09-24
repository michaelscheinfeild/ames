#!/usr/bin/env python3
"""
Quick script to inspect the HDF5 file created by extraction
"""
import h5py
import numpy as np

def inspect_hdf5(file_path):
    print(f"Inspecting: {file_path}")
    print("="*60)
    
    with h5py.File(file_path, 'r') as f:
        print("Keys in file:", list(f.keys()))
        
        if 'features' in f:
            features = f['features']
            print(f"Features shape: {features.shape}")
            print(f"Features dtype: {features.dtype}")
            
            # Check how many non-zero entries we have
            if len(features.shape) == 3:  # local features: (num_images, topk, feature_dim)
                # Count images with actual features (non-zero in the feature dimension)
                non_zero_images = 0
                for i in range(features.shape[0]):
                    if np.any(features[i]):
                        non_zero_images += 1
                    else:
                        break  # Stop at first empty image
                        
                print(f"Images with features: {non_zero_images}")
                print(f"Total allocated space for: {features.shape[0]} images")
                print(f"Features per image: {features.shape[1]}")
                print(f"Feature dimension: {features.shape[2]}")
                
                # Show some sample data from first image
                if non_zero_images > 0:
                    first_image_features = features[0]
                    print(f"First image feature range: {first_image_features.min():.4f} to {first_image_features.max():.4f}")
                    print(f"Non-zero features in first image: {np.count_nonzero(np.any(first_image_features, axis=1))}")
            
            elif len(features.shape) == 1 and len(features.dtype.names or []) > 0:  # Structured array
                print(f"Structured array with {len(features)} images")
                if 'descriptor' in features.dtype.names:
                    desc_shape = features['descriptor'][0].shape
                    print(f"Descriptor shape per image: {desc_shape}")
                if 'metadata' in features.dtype.names:
                    meta_shape = features['metadata'][0].shape
                    print(f"Metadata shape per image: {meta_shape}")
                
                # Show sample from first image
                if len(features) > 0:
                    if 'descriptor' in features.dtype.names:
                        first_desc = features['descriptor'][0]
                        print(f"First image descriptor range: {first_desc.min():.4f} to {first_desc.max():.4f}")
                    if 'metadata' in features.dtype.names:
                        first_meta = features['metadata'][0]
                        print(f"First image metadata shape: {first_meta.shape}")
                        print(f"Metadata sample: {first_meta[:5, :2]}")  # First 5 features, first 2 coords

if __name__ == "__main__":
    #inspect_hdf5("data/roxford5k/dinov2_gallery_local_500.hdf5")
    inspect_hdf5("data/roxford5k/dinov2_gallery_local.hdf5")
