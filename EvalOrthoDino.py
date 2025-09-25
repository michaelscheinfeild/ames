import sys
import os
import torch
from .tensor_dataset import TestDataset

#from src.utils.dataset_loader import get_test_loaders

'''
==================== dinov2_gallery_local.hdf5 ====================
Keys: ['features']

Dataset: features
  Shape: (20, 600, 773)
  Dtype: float32
  Size: 0.03 GB
  First image shape: (600, 773)
  First patch, first 5 features: [  0.99      0.18125   0.        0.      163.54364]
  x,y,scale,mask,attention_weight
============================================================
SUMMARY
============================================================
Based on inspection, the typical structure should be:
• Gallery features: [N_gallery, patches, 768] in HDF5
• Query features: [N_query, patches, 768] in HDF5
• Ground truth: dict with 'qimlist', 'imlist', 'gnd' keys
• Text files: Simple lists of image names
'''


pathData = r'D:\subdata\SmallData'

#test_loader = get_test_loaders(pathData, num_workers=0)
db_desc_num  = 600
test_dataset_name = 'ortho'
test_dataset_desc_dir  = 'ortho\custom'
gallery_set = TestDataset(test_dataset_name, test_dataset_desc_dir, 'dinov2_gallery_local.hdf5',
                                   desc_num=db_desc_num)
								   

print(f'Gallery set: {len(gallery_set)} samples')
#with torch.no_grad():
#    for batch in test_loader:
#        print(batch)
#        break


'''
    Expected Output
    After the script finishes, you will have a new file at this location:

    File Path: C:\OrthoPhoto\SmallData\ortho\dinov2_gallery_local.hdf5
    This HDF5 file contains the local descriptors for all your images in the Flat 
    Array format we discussed, ready to be used by the AMES model.

    Once this file is created, you can proceed to set up the evaluate.py script to
    perform retrieval tasks using your own custom dataset.

    Shape: (20, 600, 773)

    20: The number of images in your dataset.
    600: The number of local features (patches) extracted per image (--topk).
    773: The dimension of each patch's data, which is a combination of metadata and the feature vector.
    Data Type: float32 (since you are using dinov2 for non-binary features).

    Structure of the 773 Dimension
    For each of the 600 patches, the 773 values are structured as follows:

    Index Range	Content	Dimension	Description
    [0:5]	Metadata	5	Positional and validity information.
    [5:773]	Feature Vector	768	The actual DINOv2 descriptor for the patch.
    The 5 metadata values are crucial for the AMES model:

    Index	Field	Description
    0	x-coordinate	Horizontal position of the patch.
    1	y-coordinate	Vertical position of the patch.
    2	scale	Scale of the patch.
    3	mask	Validity flag (1.0=valid, 0.0=padding).
    4	attention weight	Importance score from the detector.

    For the AMES model evaluation as configured in evaluate.py,
    you primarily need the local descriptors.
    The global descriptors are often used in a full retrieval pipeline 
    for the initial search, which is then refined by AMES using the local descriptors.

    desc_type = 'cls,global,local'

    desc_type = 'local'
    # Generates: dinov2_gallery_local.hdf5

    desc_type = 'global'
    # Generates: dinov2_gallery_global.hdf5
'''