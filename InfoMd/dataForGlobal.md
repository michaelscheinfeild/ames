in github used python extract/extract_descriptors.py --dataset [gldv2|roxford5k|rparis6k]
--backbone [cvnet|dinov2]
--save_path data
--data_path [path_to_images]
--split [_gallery|_query|]
--file_name test_gallery.txt
--desc_type "local"
--pretrained , but we use file_name = 'test_query.txt' we need query and gallery is it possible or we seperate runs since in original oxford they have test_gallery.txt test_query.txt


You're absolutely right! In the original Oxford dataset setup, they have separate files for queries and gallery images, and you need separate extraction runs for each. Here's why and how to handle it:

Original Oxford Structure

roxford5k/
├── test_gallery.txt     # ~4993 gallery/database images
├── test_query.txt       # ~70 query images  
└── jpg/                 # All images

Why Separate Runs Are Needed
Different purposes:

Gallery: Database images to search in
Query: Images you search with
Different file outputs:

Gallery → dinov2_gallery_local.hdf5
Query → dinov2_query_local.hdf5
Different ground truth handling:

Query images may use bounding box cropping
Gallery images typically use full images
Solution: Two Separate Runs
# Run 1: Extract Gallery Features
# In your run_extractSingle.py - for GALLERY
if 1:  # Gallery extraction
    dataset_name = 'Ortho'
    save_path = r'C:\OrthoPhoto\data'
    data_path = r'C:\OrthoPhoto\Split'
    split = '_gallery'           # ← Gallery split
    file_name = 'test_gallery.txt'  # ← Gallery file
    desc_type = 'local,global'
    
# This creates: C:\OrthoPhoto\data\Ortho\dinov2_gallery_local.hdf5
#           and: C:\OrthoPhoto\data\Ortho\dinov2_gallery_global.hdf5


# Run 2: Extract Query Features
# In your run_extractSingle.py - for QUERY  
if 1:  # Query extraction
    dataset_name = 'Ortho'
    save_path = r'C:\OrthoPhoto\data'
    data_path = r'C:\OrthoPhoto\Split'
    split = '_query'             # ← Query split
    file_name = 'test_query.txt'    # ← Query file  
    desc_type = 'local,global'

# This creates: C:\OrthoPhoto\data\Ortho\dinov2_query_local.hdf5
#           and: C:\OrthoPhoto\data\Ortho\dinov2_query_global.hdf5
## Updated run_extractSingle.py
Here's a complete version that handles both:
'''
🎯 FEATURE EXTRACTION - GALLERY AND QUERY
Extract features for both gallery (database) and query images
'''
import os
import sys
from pathlib import Path
from extract import extract_descriptors
import gc

gc.collect()
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Add project root to Python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Configuration
dataset_name = 'Ortho'
save_path = r'C:\OrthoPhoto\data'
data_path = r'C:\OrthoPhoto\Split'
backbone = 'dinov2'
desc_type = 'local,global'
top_k = 600
num_workers = 0

# ======================================
# STEP 1: EXTRACT GALLERY FEATURES
# ======================================
print("🏛️  STEP 1: EXTRACTING GALLERY (DATABASE) FEATURES")
print("=" * 60)

gallery_args = [
    __file__,
    '--dataset', dataset_name,
    '--backbone', backbone,
    '--save_path', save_path,
    '--data_path', data_path,
    '--split', '_gallery',           # Gallery split
    '--file_name', 'test_gallery.txt',  # Gallery file
    '--desc_type', desc_type,
    '--topk', str(top_k),
    '--num_workers', str(num_workers),
    '--pretrained'
]

print(f"🔧 Gallery extraction parameters:")
print(f"   Dataset: {dataset_name}")
print(f"   Split: _gallery")
print(f"   File: test_gallery.txt")
print(f"   Output: {save_path}/{dataset_name}/dinov2_gallery_*.hdf5")

# Run gallery extraction
sys.argv = gallery_args
extract_descriptors.main()

print("✅ GALLERY EXTRACTION COMPLETE!")
print()

# ======================================  
# STEP 2: EXTRACT QUERY FEATURES
# ======================================
print("🔍 STEP 2: EXTRACTING QUERY FEATURES") 
print("=" * 60)

query_args = [
    __file__,
    '--dataset', dataset_name,
    '--backbone', backbone,
    '--save_path', save_path,
    '--data_path', data_path,
    '--split', '_query',             # Query split
    '--file_name', 'test_query.txt',    # Query file
    '--desc_type', desc_type,
    '--topk', str(top_k),
    '--num_workers', str(num_workers),
    '--pretrained'
]

print(f"🔧 Query extraction parameters:")
print(f"   Dataset: {dataset_name}")
print(f"   Split: _query") 
print(f"   File: test_query.txt")
print(f"   Output: {save_path}/{dataset_name}/dinov2_query_*.hdf5")

# Run query extraction
sys.argv = query_args
extract_descriptors.main()

print("✅ QUERY EXTRACTION COMPLETE!")
print()

# ======================================
# SUMMARY
# ======================================
print("🎉 ALL EXTRACTIONS COMPLETE!")
print("=" * 60)
print("📁 Output files created:")
print(f"   Gallery local:  {save_path}/{dataset_name}/dinov2_gallery_local.hdf5")
print(f"   Gallery global: {save_path}/{dataset_name}/dinov2_gallery_global.hdf5")  
print(f"   Query local:    {save_path}/{dataset_name}/dinov2_query_local.hdf5")
print(f"   Query global:   {save_path}/{dataset_name}/dinov2_query_global.hdf5")
print()
print("🔄 Ready for evaluation with both gallery and query features!")

# Your File Structure Should Be
C:\OrthoPhoto\Split\
├── test_gallery.txt     # List of gallery/database images
├── test_query.txt       # List of query images  
├── image1.jpg          # Gallery image
├── image2.jpg          # Gallery image
├── query1.jpg          # Query image
└── query2.jpg          # Query image

After Both Extractions
You'll have all necessary files for evaluation:

C:\OrthoPhoto\data\Ortho\
├── dinov2_gallery_local.hdf5   # Gallery local features
├── dinov2_gallery_global.hdf5  # Gallery global features
├── dinov2_query_local.hdf5     # Query local features
└── dinov2_query_global.hdf5    # Query global features

# For Evaluation
Your CrossValid.py should load query features:
# Load query features (what you search with)
local_file = r"C:\OrthoPhoto\data\Ortho\dinov2_query_local.hdf5"
global_file = r"C:\OrthoPhoto\data\Ortho\dinov2_query_global.hdf5"

Yes, you need separate runs - this is the standard approach for image retrieval datasets. Run the updated script above to extract both gallery and query features in sequence!