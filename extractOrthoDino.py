


# --- Configuration for your custom dataset ---
from unittest.mock import call
#from numpy import extract
import extract.extract_descriptors as extract_module
import sys
from pathlib import Path

# Add this to the top of the script you are executing
import os
# This environment variable disables the file locking mechanism in HDF5,
# which is a common cause of this error on Windows.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

dataset_name = 'ortho'
backbone = 'dinov2'

#save_dir = r'C:\OrthoPhoto\SmallData\ortho'   # use raw string with backslashes
#os.makedirs(save_dir, exist_ok=True)          # make sure it exists



save_path = r'C:\OrthoPhoto\SmallData'
data_path = r'C:\OrthoPhoto\SmallDb'
split = '_gallery'
file_name = 'image_list.txt' # The file you created in step 1
desc_type = 'local'
top_k = 600 # Extract 600 patches to match AMES evaluation defaults

# Simulate command-line arguments

sys.argv = [
    __file__,
    '--dataset', dataset_name,
    '--backbone', backbone,
    '--save_path', (save_path),
    '--data_path', (data_path),
    '--split', split,
    '--file_name', file_name,
    '--desc_type', desc_type,
    '--topk', str(top_k),
    '--pretrained']

print("--- Running descriptor extraction with custom settings ---")
print(f"Dataset: {dataset_name}")
print(f"Image Path: {data_path}")
print(f"Output Path: {save_path}")
print(f"Patches per image (topk): {top_k}")
print("---------------------------------------------------------")
    
#call extract\extract_descriptors.py   with my sys.argv settings
    
# Now call the main function from extract_descriptors.py
    
extract_module.main()  

'''
set PYTHONPATH=%cd%;%PYTHONPATH%

python extract\extract_descriptors.py ^
  --dataset custom ^
  --backbone dinov2 ^
  --save_path "C:\OrthoPhoto\SmallData\ortho" ^
  --data_path "C:\OrthoPhoto\SmallDb" ^
  --split _gallery ^
  --file_name "C:\OrthoPhoto\SmallData\ortho\image_list.txt" ^
  --desc_type local ^
  --topk 600 ^
  --pretrained

  python extract\extract_descriptors.py --dataset custom --backbone dinov2 --save_path "C:\OrthoPhoto\SmallData\ortho" --data_path "C:\OrthoPhoto\SmallDb" --split _gallery --file_name "C:\OrthoPhoto\SmallData\ortho\image_list.txt" --desc_type local --topk 600 --pretrained

'''