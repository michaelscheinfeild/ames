#!/usr/bin/env python3
"""
AMES Query Diagnostic Tool
Analyze what results we SHOULD get for a specific query image
"""
import pickle
import numpy as np
from pathlib import Path

def analyze_query_ground_truth(data_root, query_name="all_souls_000013"):
    """Analyze ground truth for a specific query"""
    
    data_path = Path(data_root) / "roxford5k"
    
    # Load ground truth
    with open(data_path / "gnd_roxford5k.pkl", 'rb') as f:
        gnd_data = pickle.load(f)
    
    # Load gallery names
    with open(data_path / "test_gallery.txt", 'r') as f:
        gallery_names = [line.strip() for line in f.readlines()]
    
    print("="*80)
    print(f"GROUND TRUTH ANALYSIS FOR: {query_name}")
    print("="*80)
    
    # Find query index
    query_list = gnd_data['qimlist']
    if query_name not in query_list:
        print(f"❌ Query '{query_name}' not found in query list!")
        print(f"Available queries: {query_list[:10]}...")
        return
    
    query_idx = query_list.index(query_name)
    print(f"✅ Query found at index: {query_idx}")
    
    # Get ground truth for this query
    gt_entry = gnd_data['gnd'][query_idx]
    
    print(f"\nQuery: {query_name}")
    print(f"Bounding box count: {len(gt_entry['bbx'])}")
    print(f"Easy positives: {len(gt_entry['easy'])}")
    print(f"Hard positives: {len(gt_entry['hard'])}")
    print(f"Junk images: {len(gt_entry['junk'])}")
    
    # Show expected results
    print(f"\n🎯 EXPECTED TOP RESULTS (Easy positives):")
    print("-" * 60)
    
    for i, idx in enumerate(gt_entry['easy'][:25]):  # Top 25 easy matches
        gallery_name = gnd_data['imlist'][idx]
        # Find this in our gallery
        gallery_line = None
        for j, line in enumerate(gallery_names):
            if gallery_name in line:
                gallery_line = line
                break
        
        print(f"Rank {i+1:2d}: {gallery_name:30s} | Gallery: {gallery_line}")
    
    print(f"\n🔥 HARD POSITIVES (should also match):")
    print("-" * 60)
    for i, idx in enumerate(gt_entry['hard'][:10]):  # Top 10 hard matches
        gallery_name = gnd_data['imlist'][idx]
        gallery_line = None
        for j, line in enumerate(gallery_names):
            if gallery_name in line:
                gallery_line = line
                break
        print(f"Hard {i+1:2d}: {gallery_name:30s} | Gallery: {gallery_line}")
    
    print(f"\n🗑️  JUNK (should be ignored):")
    print("-" * 60)
    for i, idx in enumerate(gt_entry['junk'][:5]):  # Show some junk
        gallery_name = gnd_data['imlist'][idx]
        print(f"Junk {i+1:2d}: {gallery_name}")
    
    # Gallery statistics
    total_gallery = len(gnd_data['imlist'])
    total_relevant = len(gt_entry['easy']) + len(gt_entry['hard'])
    
    print(f"\n📊 STATISTICS:")
    print(f"Total gallery images: {total_gallery}")
    print(f"Relevant images (easy+hard): {total_relevant}")
    print(f"Expected precision in top-25: {min(25, total_relevant)/25*100:.1f}%")
    
    return gt_entry, gnd_data

def compare_with_pipeline_results(results, gt_entry, gnd_data):
    """Compare pipeline results with ground truth"""
    
    print(f"\n🔍 PIPELINE RESULTS ANALYSIS:")
    print("="*80)
    
    easy_set = set(gnd_data['imlist'][idx] for idx in gt_entry['easy'])
    hard_set = set(gnd_data['imlist'][idx] for idx in gt_entry['hard'])
    junk_set = set(gnd_data['imlist'][idx] for idx in gt_entry['junk'])
    
    correct_easy = 0
    correct_hard = 0
    junk_found = 0
    
    print("Pipeline results vs Ground Truth:")
    print("-" * 80)
    
    for i, result in enumerate(results[:25]):
        image_name = result['image_name'].split(',')[0].replace('jpg/', '').replace('.jpg', '')
        score = result['score']
        
        status = "❌ WRONG"
        if image_name in easy_set:
            status = "✅ EASY"
            correct_easy += 1
        elif image_name in hard_set:
            status = "🔥 HARD"
            correct_hard += 1
        elif image_name in junk_set:
            status = "🗑️ JUNK"
            junk_found += 1
        
        print(f"Rank {i+1:2d}: {image_name:30s} | Score: {score:.4f} | {status}")
    
    precision = (correct_easy + correct_hard) / 25
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"Easy matches in top-25: {correct_easy}/25 ({correct_easy/25*100:.1f}%)")
    print(f"Hard matches in top-25: {correct_hard}/25 ({correct_hard/25*100:.1f}%)")
    print(f"Total relevant in top-25: {correct_easy + correct_hard}/25 ({precision*100:.1f}%)")
    print(f"Junk in top-25: {junk_found}/25 ({junk_found/25*100:.1f}%)")
    
    if precision < 0.2:  # Less than 20% precision
        print("\n⚠️  LOW PRECISION DETECTED!")
        print("Possible issues:")
        print("- Feature extraction differences")
        print("- Wrong gallery data")
        print("- Model issues")
        print("- Try a different query image")

def suggest_better_queries(data_root):
    """Suggest queries with more positive examples"""
    
    data_path = Path(data_root) / "roxford5k"
    
    with open(data_path / "gnd_roxford5k.pkl", 'rb') as f:
        gnd_data = pickle.load(f)
    
    print(f"\n🎯 BETTER QUERY SUGGESTIONS:")
    print("="*80)
    print("Queries with many easy matches (better for testing):")
    print("-" * 80)
    
    query_scores = []
    for i, (query_name, gt_entry) in enumerate(zip(gnd_data['qimlist'], gnd_data['gnd'])):
        easy_count = len(gt_entry['easy'])
        hard_count = len(gt_entry['hard'])
        total_relevant = easy_count + hard_count
        query_scores.append((query_name, easy_count, hard_count, total_relevant))
    
    # Sort by total relevant matches
    query_scores.sort(key=lambda x: x[3], reverse=True)
    
    for i, (query_name, easy, hard, total) in enumerate(query_scores[:10]):
        print(f"{i+1:2d}. {query_name:30s} | Easy: {easy:2d} | Hard: {hard:2d} | Total: {total:2d}")

if __name__ == "__main__":
    data_root = r"C:\gitRepo\ames\data"
    
    # Analyze the problematic query
    gt_entry, gnd_data = analyze_query_ground_truth(data_root, "all_souls_000013")
    
    # Suggest better queries
    suggest_better_queries(data_root)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Try a query with more easy matches (from list above)")
    print("2. Run the pipeline with a suggested query")
    print("3. Use compare_with_pipeline_results() to analyze results")
