#!/usr/bin/env python3
"""
Test AMES Pipeline with 500-image dataset
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add the parent directory to access modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import the main pipeline
from src.complete_ames_pipelinev2_fixed import CompletePipeline

def test_500_images():
    """Test the pipeline with 500 images"""
    
    print("="*80)
    print("TESTING AMES PIPELINE WITH 500 IMAGES")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        data_root=r"C:\gitRepo\ames\data", 
        model_path="dinov2_ames.pt"
    )
    
    # Override methods to use 500-image files
    original_load = pipeline.load_gallery_data
    
    def load_500_gallery_data(dataset='roxford5k'):
        """Load 500-image gallery data"""
        gallery_path = Path(pipeline.data_root) / dataset
        
        print(f"Loading 500-image gallery data...")
        
        # Load 500-image features  
        import h5py
        
        features_file = gallery_path / "dinov2_gallery_local_500.hdf5"
        with h5py.File(features_file, 'r') as f:
            data = f['features'][:]
            # Handle both old and new HDF5 formats
            if len(data.dtype.names or []) > 0:
                # Structured array format
                descriptors = data['descriptor']
            else:
                # Simple array format - take last 768 dimensions
                descriptors = data[..., -768:]
            
            pipeline.gallery_features = torch.tensor(descriptors.astype(np.float32))
            print(f"Gallery features shape: {pipeline.gallery_features.shape}")
        
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
    pipeline.load_gallery_data = load_500_gallery_data
    
    # Test with a query
    query_path = Path(pipeline.data_root) / "roxford5k" / "jpg" / "all_souls_000013.jpg"
    
    if query_path.exists():
        print(f"\nTesting with query: {query_path.name}")
        
        # Run search
        results, similarities = pipeline.search_image(str(query_path), top_k=25)
        
        print(f"\nRESULTS:")
        for i, result in enumerate(results[:10]):
            print(f"Rank {i+1:2d}: {result['image_name']:30s} | Score: {result['score']:.4f}")
        
        # Run diagnostic if available
        try:
            from src.query_diagnostic import compare_with_pipeline_results, analyze_query_ground_truth
            
            # Update data root for 500-image ground truth
            import tempfile
            original_analyze = analyze_query_ground_truth
            
            def analyze_500_ground_truth(data_root, query_name):
                # Load 500-image ground truth
                data_path = Path(data_root) / "roxford5k"
                
                import pickle
                with open(data_path / "gnd_roxford5k_500.pkl", 'rb') as f:
                    gnd_data = pickle.load(f)
                
                # Find query index
                query_list = gnd_data['qimlist']
                if query_name not in query_list:
                    print(f"Query '{query_name}' not found in 500-image dataset!")
                    return None, None
                
                query_idx = query_list.index(query_name)
                gt_entry = gnd_data['gnd'][query_idx]
                
                print(f"\n500-IMAGE GROUND TRUTH FOR: {query_name}")
                print(f"Easy positives: {len(gt_entry['easy'])}")
                print(f"Hard positives: {len(gt_entry['hard'])}")
                print(f"Junk images: {len(gt_entry['junk'])}")
                
                return gt_entry, gnd_data
            
            # Analyze ground truth
            gt_entry, gnd_data = analyze_500_ground_truth(pipeline.data_root, "all_souls_000013")
            
            if gt_entry is not None:
                # Compare results
                compare_with_pipeline_results(results, gt_entry, gnd_data)
            
        except Exception as e:
            print(f"Diagnostic failed: {e}")
        
        return results, similarities
    else:
        print(f"Query image not found: {query_path}")
        return None, None

if __name__ == "__main__":
    test_500_images()
