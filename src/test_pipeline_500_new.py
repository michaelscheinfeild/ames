#!/usr/bin/env python3
"""
Test AMES Pipeline with newly extracted 500-image features
"""
import sys
import numpy as np
import torch
import h5py
from pathlib import Path

# Add the parent directory to access modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import the main pipeline
from src.complete_ames_pipelinev2_fixed import CompletePipeline

def test_500_images_new_features():
    """Test with newly extracted 500-image features"""
    
    print("="*80)
    print("TESTING AMES PIPELINE WITH NEWLY EXTRACTED 500-IMAGE FEATURES")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        data_root=r"C:\gitRepo\ames\data", 
        model_path="dinov2_ames.pt"
    )
    
    # Override methods to use newly extracted 500-image files
    def load_new_500_gallery_data(dataset='roxford5k'):
        """Load newly extracted 500-image gallery data"""
        gallery_path = Path(pipeline.data_root) / dataset
        
        print(f"Loading newly extracted 500-image gallery data...")
        
        # Load newly extracted features (simple array format)
        features_file = gallery_path / "dinov2_gallery_500_local.hdf5"
        
        with h5py.File(features_file, 'r') as f:
            data = f['features'][:]
            print(f"New gallery features shape: {data.shape}")
            print(f"New gallery features dtype: {data.dtype}")
            
            # This should be in format (500, 700, 773) where last 768 dims are descriptors
            if len(data.shape) == 3:
                descriptors = data[..., -768:]  # Last 768 dimensions
                pipeline.gallery_features = torch.tensor(descriptors.astype(np.float32))
                print(f"✅ Extracted descriptors shape: {pipeline.gallery_features.shape}")
            else:
                print(f"❌ Unexpected data format: {data.shape}")
                return None, None
        
        # Load 500-image gallery names
        with open(gallery_path / "test_gallery_500.txt", 'r') as f:
            pipeline.gallery_names = [line.strip() for line in f.readlines()]
        
        # Load 500-image ground truth
        import pickle
        gnd_path = gallery_path / f"gnd_{dataset}_500.pkl"
        if gnd_path.exists():
            with open(gnd_path, 'rb') as f:
                pipeline.ground_truth = pickle.load(f)
                print(f"Ground truth loaded: {len(pipeline.ground_truth['qimlist'])} queries")
        
        return pipeline.gallery_features, pipeline.gallery_names
    
    # Replace the method
    pipeline.load_gallery_data = load_new_500_gallery_data
    
    # Test with a query
    query_path = Path(pipeline.data_root) / "roxford5k" / "jpg" / "all_souls_000013.jpg"
    
    if query_path.exists():
        print(f"\n🔍 Testing with query: {query_path.name}")
        
        # Run search
        results, similarities = pipeline.search_image(str(query_path), top_k=25)
        
        print(f"\n📊 RESULTS WITH NEW FEATURES:")
        for i, result in enumerate(results[:10]):
            print(f"Rank {i+1:2d}: {result['image_name']:30s} | Score: {result['score']:.4f}")
        
        # Run diagnostic
        try:
            from src.query_diagnostic import compare_with_pipeline_results
            
            # Load 500-image ground truth for diagnostic
            import pickle
            data_path = Path(pipeline.data_root) / "roxford5k"
            with open(data_path / "gnd_roxford5k_500.pkl", 'rb') as f:
                gnd_data = pickle.load(f)
            
            # Find query
            query_list = gnd_data['qimlist']
            if "all_souls_000013" in query_list:
                query_idx = query_list.index("all_souls_000013")
                gt_entry = gnd_data['gnd'][query_idx]
                
                print(f"\n📊 500-IMAGE GROUND TRUTH:")
                print(f"Easy positives: {len(gt_entry['easy'])}")
                print(f"Hard positives: {len(gt_entry['hard'])}")
                print(f"Junk images: {len(gt_entry['junk'])}")
                
                # Compare results
                compare_with_pipeline_results(results, gt_entry, gnd_data)
            
        except Exception as e:
            print(f"Diagnostic failed: {e}")
        
        return results, similarities
    else:
        print(f"❌ Query image not found: {query_path}")
        return None, None

if __name__ == "__main__":
    test_500_images_new_features()
