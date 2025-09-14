#!/usr/bin/env python3
"""
Re-extract gallery features using the same pipeline as your search
This should fix the mismatch between gallery features and pipeline features
"""
import os
import sys
from pathlib import Path
import torch
import h5py
import numpy as np
from tqdm import tqdm

# Add the parent directory to access the pipeline
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import your pipeline
from complete_ames_pipelinev2_fixed import CompletePipeline

def re_extract_gallery_features(data_root):
    """Re-extract gallery features using the exact same pipeline as search"""
    
    print("="*80)
    print("RE-EXTRACTING GALLERY FEATURES")
    print("="*80)
    
    # Initialize the same pipeline used for search
    pipeline = CompletePipeline(data_root=data_root, model_path="dinov2_ames.pt")
    
    # Load gallery image names
    gallery_path = Path(data_root) / "roxford5k"
    with open(gallery_path / "test_gallery.txt", 'r') as f:
        gallery_names = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(gallery_names)} gallery images")
    
    # Extract features for all gallery images
    all_features = []
    
    for i, gallery_name in enumerate(tqdm(gallery_names, desc="Extracting features")):
        # Extract base image name and construct path
        base_name = gallery_name.split(',')[0]  # Remove metadata
        image_path = gallery_path / base_name
        
        try:
            # Use the same feature extraction as in search
            features = pipeline.extract_ames_style_features(str(image_path), topk=700)
            
            # Extract only the descriptor part (last 768 dims) to match gallery format
            descriptor = features[:, -768:].numpy().astype(np.float16)  # Match original dtype
            
            # Create metadata (dummy values matching original format)
            metadata = np.zeros((700, 5), dtype=np.float32)
            metadata[:, 0] = features[:, 0].numpy()  # x
            metadata[:, 1] = features[:, 1].numpy()  # y  
            metadata[:, 2] = features[:, 2].numpy()  # scale_enc
            metadata[:, 3] = features[:, 3].numpy()  # mask
            metadata[:, 4] = features[:, 4].numpy()  # weights
            
            # Store in same format as original
            feature_entry = np.zeros(1, dtype=[('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))])
            feature_entry['metadata'][0] = metadata
            feature_entry['descriptor'][0] = descriptor
            
            all_features.append(feature_entry[0])
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(gallery_names)} images")
                
        except Exception as e:
            print(f"Error processing {gallery_name}: {e}")
            # Create dummy entry to maintain indexing
            feature_entry = np.zeros(1, dtype=[('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))])
            all_features.append(feature_entry[0])
    
    # Save the re-extracted features
    all_features = np.array(all_features)
    output_path = gallery_path / "dinov2_gallery_local_reextracted.hdf5"
    
    print(f"Saving re-extracted features to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=all_features)
    
    print(f"✅ Re-extracted {len(all_features)} gallery features")
    print(f"✅ Saved to: {output_path}")
    
    return output_path

def test_with_reextracted_features(data_root, reextracted_path):
    """Test the pipeline with re-extracted features"""
    
    print("\n" + "="*80)
    print("TESTING WITH RE-EXTRACTED FEATURES")
    print("="*80)
    
    # Modify the pipeline to use re-extracted features
    pipeline = CompletePipeline(data_root=data_root, model_path="dinov2_ames.pt")
    
    # Override the gallery loading to use re-extracted features
    original_load = pipeline.load_gallery_data
    
    def load_reextracted_gallery(dataset='roxford5k'):
        gallery_path = Path(data_root) / dataset
        
        print(f"Loading RE-EXTRACTED gallery data from {reextracted_path}...")
        
        # Load re-extracted features
        with h5py.File(reextracted_path, 'r') as f:
            data = f['features'][:]
            descriptors = data['descriptor']
            pipeline.gallery_features = torch.tensor(descriptors.astype(np.float32))
            print(f"Re-extracted gallery features shape: {pipeline.gallery_features.shape}")
        
        # Load gallery names
        with open(gallery_path / "test_gallery.txt", 'r') as f:
            pipeline.gallery_names = [line.strip() for line in f.readlines()]
        
        # Load ground truth
        import pickle
        gnd_path = gallery_path / f"gnd_{dataset}.pkl"
        if gnd_path.exists():
            with open(gnd_path, 'rb') as f:
                pipeline.ground_truth = pickle.load(f)
        
        return pipeline.gallery_features, pipeline.gallery_names
    
    # Replace the method
    pipeline.load_gallery_data = load_reextracted_gallery
    
    # Test with a good query
    query_path = Path(data_root) / "roxford5k" / "jpg" / "all_souls_000013.jpg"
    
    print(f"Testing with query: {query_path.name}")
    
    # Run search
    results, similarities = pipeline.search_image(str(query_path), top_k=25)
    
    return results, similarities

if __name__ == "__main__":
    data_root = r"C:\gitRepo\ames\data"
    
    print("🔧 This will re-extract gallery features using your exact pipeline")
    print("⚠️  This may take a while (4993 images)")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        exit()
    
    # Re-extract features
    reextracted_path = re_extract_gallery_features(data_root)
    
    # Test with re-extracted features
    print("\n🧪 Testing with re-extracted features...")
    results, similarities = test_with_reextracted_features(data_root, reextracted_path)
    
    print("✅ Testing complete - check the results!")
