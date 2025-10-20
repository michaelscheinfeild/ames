import os
import pickle
import re
import numpy as np
from typing import List, Dict, Tuple

def extract_coordinates_from_filename(filename):
    """Extract x, y coordinates from orthophoto filename"""
    try:
        base_name = os.path.splitext(filename)[0]
        match = re.search(r'(\d+)_(\d+)', base_name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return x, y
        return None, None
    except Exception as e:
        print(f"❌ Error extracting coordinates from {filename}: {e}")
        return None, None

def calculate_overlap_percentage(coord1, coord2, tile_width=1120, tile_height=700):
    """Calculate overlap percentage between two orthophoto tiles"""
    if coord1[0] is None or coord1[1] is None or coord2[0] is None or coord2[1] is None:
        return 0.0
    
    x1, y1 = coord1
    x2, y2 = coord2
    
    # Calculate tile boundaries (assuming coordinates are top-left corners)
    tile1_left, tile1_right = x1, x1 + tile_width
    tile1_top, tile1_bottom = y1, y1 + tile_height
    
    tile2_left, tile2_right = x2, x2 + tile_width  
    tile2_top, tile2_bottom = y2, y2 + tile_height
    
    # Calculate intersection
    intersection_left = max(tile1_left, tile2_left)
    intersection_right = min(tile1_right, tile2_right)
    intersection_top = max(tile1_top, tile2_top)
    intersection_bottom = min(tile1_bottom, tile2_bottom)
    
    # Check if there's actual overlap
    if intersection_left >= intersection_right or intersection_top >= intersection_bottom:
        return 0.0
    
    # Calculate overlap area
    intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
    tile_area = tile_width * tile_height
    
    # Overlap percentage relative to tile size
    overlap_percentage = intersection_area / tile_area
    return overlap_percentage

def create_orthophoto_gnd(data_path: str, image_list_file: str, 
                         overlap_threshold_easy=0.25, overlap_threshold_hard=0.15,
                         tile_width=1120, tile_height=700, include_self=True):
    """
    Create ground truth file for orthophoto dataset based on spatial overlap
    
    Args:
        data_path: Path to dataset folder
        image_list_file: Name of file containing image list (e.g., 'single_image.txt')
        overlap_threshold_easy: Threshold for easy positive matches (default 0.25 = 25%)
        overlap_threshold_hard: Threshold for hard positive matches (default 0.15 = 15%)
        tile_width, tile_height: Orthophoto tile dimensions
        include_self: Whether to include self-matches in easy category (default True)
    
    Note: For orthophotos, typical overlap is 20-30% between adjacent tiles
    """
    
    print(f"🔧 Creating orthophoto ground truth file...")
    print(f"📁 Data path: {data_path}")
    print(f"📄 Image list: {image_list_file}")
    print(f"🎯 Easy threshold: {overlap_threshold_easy * 100:.1f}% (includes self-matches)")
    print(f"🎯 Hard threshold: {overlap_threshold_hard * 100:.1f}%")
    print(f"🔄 Include self-matches: {include_self}")
    
    # Read image list
    image_list_path = os.path.join(data_path, image_list_file)
    if not os.path.exists(image_list_path):
        raise FileNotFoundError(f"Image list file not found: {image_list_path}")
    
    # Parse image names from the file
    image_names = []
    with open(image_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Handle different formats: "path,query_id,width,height" or just "filename"
                if ',' in line:
                    filename = line.split(',')[0]
                    image_names.append(os.path.basename(filename))
                else:
                    image_names.append(os.path.basename(line))
    
    print(f"📊 Found {len(image_names)} images")
    
    # Extract coordinates for all images
    image_coords = []
    valid_images = []
    
    for i, img_name in enumerate(image_names):
        coords = extract_coordinates_from_filename(img_name)
        if coords[0] is not None and coords[1] is not None:
            image_coords.append(coords)
            valid_images.append((i, img_name))
        else:
            print(f"⚠️ Skipping image with invalid coordinates: {img_name}")
    
    print(f"✅ Valid images with coordinates: {len(valid_images)}")
    
    # Analyze overlap patterns first
    print(f"\n🔍 Analyzing overlap patterns...")
    all_overlaps = []
    
    for i in range(len(valid_images)):
        for j in range(i + 1, len(valid_images)):  # Only upper triangle to avoid duplicates
            coord1 = image_coords[i]
            coord2 = image_coords[j]
            overlap = calculate_overlap_percentage(coord1, coord2, tile_width, tile_height)
            if overlap > 0:
                all_overlaps.append(overlap)
                name1 = valid_images[i][1]
                name2 = valid_images[j][1]
                print(f"  {name1} ↔ {name2}: {overlap:.1%} overlap")
    
    if all_overlaps:
        print(f"\n📊 Overlap Statistics:")
        print(f"   Non-zero overlaps found: {len(all_overlaps)}")
        print(f"   Min overlap: {min(all_overlaps):.1%}")
        print(f"   Max overlap: {max(all_overlaps):.1%}")
        print(f"   Mean overlap: {np.mean(all_overlaps):.1%}")
        print(f"   Median overlap: {np.median(all_overlaps):.1%}")
    else:
        print(f"⚠️ No overlapping tiles found! Check coordinates and tile dimensions.")
    
    # Create ground truth for each image as query
    gnd_entries = []
    
    for query_idx, (orig_idx, query_name) in enumerate(valid_images):
        query_coords = image_coords[query_idx]
        
        # Collect all matches with their overlaps
        all_matches = []
        
        # Compare with ALL images (including self)
        for db_idx, (db_orig_idx, db_name) in enumerate(valid_images):
            db_coords = image_coords[db_idx]
            
            if query_idx == db_idx:
                # Self-match: 100% overlap
                overlap = 1.0
            else:
                # Calculate actual overlap
                overlap = calculate_overlap_percentage(query_coords, db_coords, tile_width, tile_height)
            
            if overlap > 0:  # Any overlap
                all_matches.append((db_idx, overlap, db_name))
        
        # Sort by overlap (highest first)
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize matches
        easy_matches = []
        hard_matches = []
        junk_matches = []
        
        for db_idx, overlap, db_name in all_matches:
            if overlap >= overlap_threshold_hard:  # Above minimum threshold
                hard_matches.append(db_idx)
                
                if overlap >= overlap_threshold_easy:  # High overlap (includes self)
                    easy_matches.append(db_idx)
        
        # Create ground truth entry
        gnd_entry = {
            'easy': easy_matches,    # High overlap matches (includes self at 100%)
            'hard': hard_matches,    # All relevant matches (superset of easy)
            'junk': junk_matches,    # Empty for orthophotos
            'bbx': [0, 0, tile_width, tile_height]  # Full tile as query region
        }
        
        gnd_entries.append(gnd_entry)
        
        # Show detailed results with names of matches
        self_included = query_idx in easy_matches
        print(f"Query {query_idx+1:2d} ({query_name}):")
        print(f"  📊 Total relevant: {len(hard_matches)}, Easy: {len(easy_matches)}")
        print(f"  🔄 Self-match included: {self_included}")
        
        # Show names of easy matches
        if easy_matches:
            easy_names = [valid_images[idx][1] for idx in easy_matches]
            print(f"  ✅ Easy matches: {easy_names}")
        
        # Show names of hard matches (non-easy ones)
        hard_only = [idx for idx in hard_matches if idx not in easy_matches]
        if hard_only:
            hard_only_names = [valid_images[idx][1] for idx in hard_only]
            print(f"  🔶 Hard-only matches: {hard_only_names}")
        
        # Show overlap percentages for all matches
        if all_matches and len(all_matches) > 1:
            print(f"  📏 All overlaps:")
            for db_idx, overlap, db_name in all_matches:
                if db_idx == query_idx:
                    print(f"    - {db_name}: {overlap:.1%} (SELF)")
                else:
                    category = "Easy" if db_idx in easy_matches else "Hard" if db_idx in hard_matches else "Below threshold"
                    print(f"    - {db_name}: {overlap:.1%} ({category})")
    
    # Validate ground truth structure
    print(f"\n🔍 Validating ground truth structure...")
    valid_structure = True
    
    for i, gnd_entry in enumerate(gnd_entries):
        easy_set = set(gnd_entry['easy'])
        hard_set = set(gnd_entry['hard'])
        
        # Check that easy ⊆ hard
        if not easy_set.issubset(hard_set):
            print(f"❌ Query {i}: Easy matches not subset of hard matches!")
            valid_structure = False
        
        # Check that self-match is included in easy (if include_self=True)
        if include_self and i not in easy_set:
            print(f"❌ Query {i}: Self-match not in easy matches!")
            valid_structure = False
    
    if valid_structure:
        print(f"✅ Ground truth structure is valid")
    else:
        print(f"❌ Ground truth structure has errors!")
        return None
    
    # Create the complete ground truth structure
    gnd_data = {
        'qimlist': [name for _, name in valid_images],  # Query images
        'imlist': [name for _, name in valid_images],   # Database images (same as queries)
        'gnd': gnd_entries
    }
    
    # Save ground truth file
    gnd_filename = f'gnd_orthophoto.pkl'
    gnd_path = os.path.join(data_path, gnd_filename)
    
    with open(gnd_path, 'wb') as f:
        pickle.dump(gnd_data, f)
    
    print(f"\n✅ Ground truth saved: {gnd_path}")
    print(f"📊 Final Statistics:")
    print(f"   Total queries: {len(gnd_entries)}")
    print(f"   Average total relevant per query: {np.mean([len(g['hard']) for g in gnd_entries]):.1f}")
    print(f"   Average easy matches per query: {np.mean([len(g['easy']) for g in gnd_entries]):.1f}")
    
    # Show some examples
    print(f"\n📋 Example ground truth entries:")
    for i in range(min(3, len(gnd_entries))):
        query_name = gnd_data['qimlist'][i]
        entry = gnd_entries[i]
        print(f"  Query {i}: {query_name}")
        print(f"    Easy: {entry['easy']} (includes self: {i in entry['easy']})")
        print(f"    Hard: {entry['hard']}")
    
    # Show detailed examples with names
    print(f"\n📋 Detailed Ground Truth Examples:")
    for i in range(min(3, len(gnd_entries))):
        query_name = gnd_data['qimlist'][i]
        entry = gnd_entries[i]
        
        print(f"\n  Query {i}: {query_name}")
        
        # Show easy match names
        if entry['easy']:
            easy_names = [gnd_data['imlist'][idx] for idx in entry['easy']]
            print(f"    ✅ Easy matches ({len(entry['easy'])}): {easy_names}")
        else:
            print(f"    ✅ Easy matches: None")
        
        # Show hard match names  
        if entry['hard']:
            hard_names = [gnd_data['imlist'][idx] for idx in entry['hard']]
            print(f"    🔶 Hard matches ({len(entry['hard'])}): {hard_names}")
        else:
            print(f"    🔶 Hard matches: None")
        
        # Show hard-only matches (hard but not easy)
        hard_only = [idx for idx in entry['hard'] if idx not in entry['easy']]
        if hard_only:
            hard_only_names = [gnd_data['imlist'][idx] for idx in hard_only]
            print(f"    📊 Hard-only matches: {hard_only_names}")
        
        # Verify subset relationship
        easy_subset = set(entry['easy']).issubset(set(entry['hard']))
        print(f"    ✓ Easy ⊆ Hard: {easy_subset}")
    
    return gnd_path

if __name__ == "__main__":
    # Updated configuration for realistic orthophoto overlaps
    data_path = r'C:\OrthoPhoto\Split'
    image_list_file = 'single_image.txt'
    
    gnd_path = create_orthophoto_gnd(
        data_path=data_path,
        image_list_file=image_list_file,
        overlap_threshold_easy=0.25,   # 25% overlap = easy match (typical orthophoto overlap)
        overlap_threshold_hard=0.15,   # 15% overlap = minimum relevance
        tile_width=1120,
        tile_height=700,
        include_self=True              # Include self-matches in easy category
    )
    
    if gnd_path:
        print(f"\n🎉 Orthophoto ground truth created: {gnd_path}")
    else:
        print(f"\n❌ Failed to create ground truth file")