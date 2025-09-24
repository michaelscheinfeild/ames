#!/usr/bin/env python3
"""
Setup 500-image dataset for AMES pipeline testing
"""
import os
import shutil
import pickle
import h5py
import numpy as np
from pathlib import Path

def create_500_image_dataset(data_root):
    """Create all necessary files for 500-image testing"""
    
    print("="*80)
    print("CREATING 500-IMAGE DATASET FOR AMES TESTING")
    print("="*80)
    
    data_path = Path(data_root) / "roxford5k"
    
    # 1. Copy the existing 500-image features file
    src_features = data_path / "dinov2_gallery_local.hdf5"
    dst_features = data_path / "dinov2_gallery_local_500.hdf5"
    
    if src_features.exists():
        print(f"✅ Copying {src_features} -> {dst_features}")
        shutil.copy2(src_features, dst_features)
    else:
        print(f"❌ Source features file not found: {src_features}")
        return
    
    # 2. Create test_gallery_500.txt (already exists, but verify)
    gallery_500_file = data_path / "test_gallery_500.txt"
    if not gallery_500_file.exists():
        # Create it from the main gallery file
        with open(data_path / "test_gallery.txt", 'r') as f:
            lines = f.readlines()[:500]
        
        with open(gallery_500_file, 'w') as f:
            f.writelines(lines)
        print(f"✅ Created {gallery_500_file}")
    else:
        print(f"✅ {gallery_500_file} already exists")
    
    # 3. Load original ground truth and create 500-image version
    gnd_file = data_path / "gnd_roxford5k.pkl"
    gnd_500_file = data_path / "gnd_roxford5k_500.pkl"
    
    if gnd_file.exists():
        print(f"📊 Processing ground truth data...")
        
        with open(gnd_file, 'rb') as f:
            original_gnd = pickle.load(f)
        
        # Load gallery names for mapping
        with open(data_path / "test_gallery.txt", 'r') as f:
            full_gallery = [line.strip().split(',')[0].replace('jpg/', '') for line in f.readlines()]
        
        with open(gallery_500_file, 'r') as f:
            gallery_500 = [line.strip().split(',')[0].replace('jpg/', '') for line in f.readlines()]
        
        # Create mapping from full gallery index to 500-image index
        gallery_500_set = set(gallery_500)
        old_to_new_idx = {}
        
        for old_idx, img_name in enumerate(full_gallery):
            if img_name in gallery_500_set:
                new_idx = gallery_500.index(img_name)
                old_to_new_idx[old_idx] = new_idx
        
        print(f"   Mapped {len(old_to_new_idx)} images from full dataset to 500-image subset")
        
        # Process ground truth entries
        new_gnd_data = {
            'qimlist': original_gnd['qimlist'].copy(),
            'imlist': [original_gnd['imlist'][i] for i in range(500)],  # First 500 images
            'gnd': []
        }
        
        for i, gt_entry in enumerate(original_gnd['gnd']):
            # Map indices to new 500-image dataset
            new_easy = [old_to_new_idx[idx] for idx in gt_entry['easy'] if idx in old_to_new_idx]
            new_hard = [old_to_new_idx[idx] for idx in gt_entry['hard'] if idx in old_to_new_idx]
            new_junk = [old_to_new_idx[idx] for idx in gt_entry['junk'] if idx in old_to_new_idx]
            
            new_entry = {
                'easy': np.array(new_easy, dtype=np.int32),
                'hard': np.array(new_hard, dtype=np.int32), 
                'junk': np.array(new_junk, dtype=np.int32),
                'bbx': gt_entry['bbx'].copy()
            }
            new_gnd_data['gnd'].append(new_entry)
            
            if i < 5:  # Show first few queries
                query_name = original_gnd['qimlist'][i]
                print(f"   Query {i+1} ({query_name}): {len(new_easy)} easy, {len(new_hard)} hard, {len(new_junk)} junk")
        
        # Save new ground truth
        with open(gnd_500_file, 'wb') as f:
            pickle.dump(new_gnd_data, f)
        
        print(f"✅ Created {gnd_500_file}")
        print(f"   Queries: {len(new_gnd_data['qimlist'])}")
        print(f"   Gallery: {len(new_gnd_data['imlist'])}")
    
    # 4. Create query features for 500-image dataset (extract a few queries)
    print(f"\n🔍 Creating query features for testing...")
    
    # Copy some existing query features or create placeholder
    query_src = data_path / "dinov2_query_local.hdf5"
    query_dst = data_path / "dinov2_query_local_500.hdf5"
    
    if query_src.exists():
        shutil.copy2(query_src, query_dst)
        print(f"✅ Copied query features: {query_dst}")
    else:
        print(f"⚠️  No existing query features found")
    
    # 5. Verify the created files
    print(f"\n📋 VERIFICATION:")
    print(f"✅ Gallery features: {dst_features} ({dst_features.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"✅ Gallery list: {gallery_500_file} ({len(open(gallery_500_file).readlines())} images)")
    print(f"✅ Ground truth: {gnd_500_file}")
    
    if query_dst.exists():
        print(f"✅ Query features: {query_dst}")
    
    return dst_features, gallery_500_file, gnd_500_file

def modify_pipeline_for_500_images():
    """Create a modified pipeline script for 500-image testing"""
    
    pipeline_500_content = '''#!/usr/bin/env python3
"""
Complete AMES Pipeline for 500-image testing
Modified to use 500-image dataset files
"""
import sys
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
    
    # Initialize pipeline with 500-image dataset
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
        import torch
        
        features_file = gallery_path / "dinov2_gallery_local_500.hdf5"
        with h5py.File(features_file, 'r') as f:
            data = f['features'][:]
            descriptors = data['descriptor'] if 'descriptor' in data.dtype.names else data[..., -768:]
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
        print(f"\\n🔍 Testing with query: {query_path.name}")
        
        # Run search
        results, similarities = pipeline.search_image(str(query_path), top_k=25)
        
        print(f"\\n📊 RESULTS:")
        for i, result in enumerate(results[:10]):
            print(f"Rank {i+1:2d}: {result['image_name']:30s} | Score: {result['score']:.4f}")
        
        # Run diagnostic if available
        try:
            from src.query_diagnostic import compare_with_pipeline_results, analyze_query_ground_truth
            
            # Analyze ground truth
            gt_entry, gnd_data = analyze_query_ground_truth(pipeline.data_root, "all_souls_000013")
            
            # Compare results
            compare_with_pipeline_results(results, gt_entry, gnd_data)
            
        except Exception as e:
            print(f"Diagnostic failed: {e}")
        
        return results, similarities
    else:
        print(f"❌ Query image not found: {query_path}")
        return None, None

if __name__ == "__main__":
    test_500_images()
'''
    
    # Save the 500-image pipeline script
    script_path = Path("src/test_pipeline_500.py")
    with open(script_path, 'w') as f:
        f.write(pipeline_500_content)
    
    print(f"✅ Created 500-image pipeline script: {script_path}")
    return script_path

if __name__ == "__main__":
    data_root = r"C:\gitRepo\ames\data"
    
    # Create 500-image dataset
    features_file, gallery_file, gnd_file = create_500_image_dataset(data_root)
    
    # Create modified pipeline script
    pipeline_script = modify_pipeline_for_500_images()
    
    print(f"\n🚀 READY TO TEST!")
    print(f"Run: python {pipeline_script}")
