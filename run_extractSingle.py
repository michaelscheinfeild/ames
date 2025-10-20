'''

    🎯 Summary of What We're Doing
    -Extract features for N images: all_souls_000013.jpg ...
    -Use DINOv2 backbone: Non-binary, 768-dim features
    -Local descriptors only: No global features needed
    -600 patches per image: Good balance of detail vs speed
    -Save to standard location: Compatible with AMES evaluation
'''
import os
import sys
from pathlib import Path

from extract import extract_descriptors

import gc

gc.collect()

# Disable HDF5 file locking for Windows
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'



def get_WorkSpaceVars(dataset_name, workLocation):
    # --- Configuration for single Oxford image extraction ---
    if dataset_name == 'roxford5k':
    
        file_name = 'test_query_100.txt'  # The file we created with just one image

        if workLocation == 'Work':
                save_path = r'C:\github\ames\ames\data'
                data_path = r'C:\github\ames\ames\data\roxford5k'  # Root path where jpg folder is
                
        else:
                save_path = r'C:\gitRepo\ames\data'
                data_path = r'C:\gitRepo\ames\data\roxford5k'  # Root path where jpg folder is
        
        
    else: # OrthoPhoto dataset

            save_path = r'C:\OrthoPhoto\data'
            data_path = r'C:\OrthoPhoto\Split'  # Root path where jpg folder is
            file_name = 'test_query.txt'  # The file we created with just one image


    return save_path, data_path, file_name


#-------------------------------------------------
# Add project root to Python path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))    

dataSetTypes = ['roxford5k', 'Ortho']
workTypesLocation =['Home','Work']

dataset_name = dataSetTypes[1]  # Change to 'Ortho' for OrthoPhoto dataset
workLocation = workTypesLocation[0]  # Change to 'Work' if running at work  

split = '_query'  # Using query split since we're processing a query image
save_path, data_path, file_name = get_WorkSpaceVars(dataset_name, workLocation)

print(f"Dataset: {dataset_name}")
print(f"Work Location: {workLocation}")


backbone = 'dinov2'
#desc_type = 'local'  # Only local features
desc_type = 'local,global'
top_k = 600  # Extract 600 patches
num_workers = 0  # Set to 0 for Windows to avoid issues with multiprocessing

# we run twice query and gallery    
# galley
if 0:
    file_name = 'test_gallery.txt'  # Change to gallery for OrthoPhoto dataset
    split = '_gallery'  # Change to gallery for OrthoPhoto dataset

if 1:
    file_name = 'test_query.txt'  # Change to gallery for OrthoPhoto dataset
    split = '_query'  # Change to gallery for OrthoPhoto dataset

# Simulate command-line arguments
sys.argv = [
    __file__,
    '--dataset', dataset_name,
    '--backbone', backbone,
    '--save_path', save_path,
    '--data_path', data_path,
    '--split', split,
    '--file_name', file_name,
    '--desc_type', desc_type,
    '--topk', str(top_k),
    '--num_workers', str(num_workers),
    #'--batch_size', '16',  
    '--pretrained'  # Use pretrained detector
]

print("🔧 SINGLE IMAGE EXTRACTION - Running with these parameters:")
print(f"Dataset: {dataset_name}")
print(f"Backbone: {backbone} (non-binary)")
print(f"Data path: {data_path}")
print(f"Save path: {save_path}")
print(f"Image file: {file_name}")
print(f"Split: {split}")
print(f"Descriptor type: {desc_type}")
print(f"Top-k patches: {top_k}")
print("=" * 60)

# Call the main function
extract_descriptors.main()  # ← Add .main() here

print("\n🎉 EXTRACTION COMPLETE!")
print(f"✅ Output file: {save_path}/{dataset_name}/dinov2{split}_local.hdf5")
print(f"📊 Expected shape: (Nimg, {top_k}, 773)")
print("📋 Data format: [1 image, 600 patches, 5 metadata + 768 features]")

'''
🎉 EXTRACTION COMPLETE!
✅ Output file: C:\gitRepo\ames\data/roxford5k/dinov2_query_local.hdf5
📊 Expected shape: (Nimg, 600, 773)
📋 Data format: [Nimg image, 600 patches, 5 metadata + 768 features]



use local,global
📊 Expected shape: (Nimg, 600, 773)
📋 Data format: [Nimg image, 600 patches, 5 metadata + 768 features]


 29/9
🎉 EXTRACTION COMPLETE!
✅ Output file: C:\github\ames\ames\data/roxford5k/dinov2_query_local.hdf5
📊 Expected shape: (Nimg, 600, 773)
📋 Data format: [1 image, 600 patches, 5 metadata + 768 features]


 8 images ortho
 
🎉 EXTRACTION COMPLETE!
✅ Output file: C:\OrthoPhoto\data/Ortho/dinov2_query_local.hdf5
📊 Expected shape: (Nimg, 600, 773)
📋 Data format: [1 image, 600 patches, 5 metadata + 768 features]


20/10
✅ Output file: C:\OrthoPhoto\data/Ortho/dinov2_gallery_local.hdf5
📊 Expected shape: (Nimg, 600, 773)
📋 Data format: [1 image, 600 patches, 5 metadata + 768 features]
'''