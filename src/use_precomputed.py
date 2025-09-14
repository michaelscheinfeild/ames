#!/usr/bin/env python3
"""
Use pre-computed nearest neighbors instead of running AMES inference
This should give you the correct results that match the gallery features
"""
import pickle
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def load_precomputed_results(data_root, query_name="all_souls_000013"):
    """Load pre-computed nearest neighbor results"""
    
    data_path = Path(data_root) / "roxford5k"
    
    # Load ground truth to get query index
    with open(data_path / "gnd_roxford5k.pkl", 'rb') as f:
        gnd_data = pickle.load(f)
    
    # Load gallery names
    with open(data_path / "test_gallery.txt", 'r') as f:
        gallery_names = [line.strip() for line in f.readlines()]
    
    # Find query index
    query_list = gnd_data['qimlist']
    if query_name not in query_list:
        print(f"❌ Query '{query_name}' not found!")
        return None
        
    query_idx = query_list.index(query_name)
    print(f"✅ Query '{query_name}' found at index: {query_idx}")
    
    # Load pre-computed nearest neighbors
    with open(data_path / "nn_dinov2.pkl", 'rb') as f:
        nn_data = pickle.load(f)
    
    print(f"Pre-computed NN data type: {type(nn_data)}")
    print(f"Pre-computed NN data shape: {nn_data.shape if hasattr(nn_data, 'shape') else 'No shape'}")
    
    # Get nearest neighbors for this query
    if isinstance(nn_data, torch.Tensor):
        # This appears to be similarity scores, not indices
        # We need to get the top indices by sorting
        query_similarities = nn_data[0, query_idx, :]  # Take first dimension, query index, all gallery
        print(f"Query similarities shape: {query_similarities.shape}")
        print(f"Sample similarities: {query_similarities[:10]}")
        
        # Get top 25 indices by sorting similarities in descending order
        sorted_indices = torch.argsort(query_similarities, descending=True)
        top_indices = sorted_indices[:25]
        top_scores = query_similarities[top_indices]
        
        print(f"Top 10 indices: {top_indices[:10]}")
        print(f"Top 10 scores: {top_scores[:10]}")
    else:
        print("❌ Unexpected data format")
        return None
    
    # Create results list
    results = []
    for rank, (gallery_idx, score) in enumerate(zip(top_indices[:25], top_scores[:25])):
        gallery_idx = int(gallery_idx.item())
        score = float(score.item())
        result = {
            'rank': rank + 1,
            'index': gallery_idx,
            'score': score,
            'image_name': gallery_names[gallery_idx] if gallery_idx < len(gallery_names) else f"image_{gallery_idx}"
        }
        results.append(result)
    
    return results, gnd_data

def visualize_precomputed_results(results, data_root, query_name):
    """Visualize the pre-computed results"""
    
    data_path = Path(data_root) / "roxford5k"
    query_image_path = data_path / "jpg" / f"{query_name}.jpg"
    
    # Create figure
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle('Pre-computed AMES Results - Top 25', fontsize=16, fontweight='bold')
    
    # Load and display query image
    query_img = cv2.imread(str(query_image_path))
    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Query image subplot
    ax_query = plt.subplot(6, 5, (1, 5))  # Top row, spans all 5 columns
    ax_query.imshow(query_img_rgb)
    ax_query.set_title(f'Query: {query_name}.jpg', fontsize=12, fontweight='bold', pad=15)
    ax_query.axis('off')
    
    # Display results
    for i, result in enumerate(results[:25]):
        # Extract base image name
        base_name = result['image_name'].split(',')[0]
        result_image_path = data_path / base_name
        
        try:
            # Load result image
            result_img = cv2.imread(str(result_image_path))
            if result_img is not None:
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # Calculate subplot position
                row = 2 + (i // 5)
                col = (i % 5) + 1
                
                ax = plt.subplot(6, 5, (row - 1) * 5 + col)
                ax.imshow(result_img_rgb)
                ax.set_title(f'Rank {result["rank"]}\n{Path(base_name).stem}\nScore: {result["score"]:.3f}', 
                            fontsize=7, pad=3)
                ax.axis('off')
            else:
                print(f"Could not load: {result_image_path}")
                
        except Exception as e:
            print(f"Error loading {result['image_name']}: {e}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_root = r"C:\gitRepo\ames\data"
    query_name = "all_souls_000013"
    
    print("="*80)
    print("USING PRE-COMPUTED RESULTS")
    print("="*80)
    
    # Load pre-computed results
    results, gnd_data = load_precomputed_results(data_root, query_name)
    
    if results:
        print(f"\nTop 25 pre-computed results:")
        print("-" * 60)
        for result in results:
            print(f"Rank {result['rank']:2d}: {result['image_name']:40s} | Score: {result['score']:.4f}")
        
        # Compare with ground truth
        try:
            from query_diagnostic import compare_with_pipeline_results, analyze_query_ground_truth
            gt_entry, _ = analyze_query_ground_truth(data_root, query_name)
            compare_with_pipeline_results(results, gt_entry, gnd_data)
        except:
            print("Could not run ground truth comparison")
        
        # Visualize
        visualize_precomputed_results(results, data_root, query_name)
    else:
        print("❌ Could not load pre-computed results")
