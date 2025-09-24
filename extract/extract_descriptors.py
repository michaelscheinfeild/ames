import argparse
import os
import pickle
from functools import partial

import torch
from torch.utils.data import SequentialSampler, BatchSampler


from .extract_dino import extract as extract_dino, load_dinov2
from .extract_cvnet import extract as extract_cvnet, load_cvnet
from .image_dataset import read_imlist, DataSet, FeatureStorage
from .spatial_attention_2d import SpatialAttention2d
'''
from extract_dino import extract as extract_dino, load_dinov2
from extract_cvnet import extract as extract_cvnet, load_cvnet
from image_dataset import read_imlist, DataSet, FeatureStorage
from spatial_attention_2d import SpatialAttention2d
'''

import sys

_BASE_URL = 'http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks/'


def main():
    parser = argparse.ArgumentParser(description='Generate 1M embedding')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Load pretrained detector model.')
    parser.add_argument('--save_path', default='data', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--file_name', help='file name to parse image paths')
    parser.add_argument('--dataset', help='dataset')
    parser.add_argument('--split', nargs='?', const='', default='', type=str)
    parser.add_argument('--desc_type', default='cls,global,local', type=str)
    parser.add_argument('--backbone', default='dinov2', type=str)
    parser.add_argument('--topk', default=700, type=int)
    parser.add_argument('--imsize', type=int)
    parser.add_argument('--num_workers', default=8, type=int)


    args = parser.parse_args()
    dataset, file_name, imsize, topk, desc_type = args.dataset, args.file_name, args.imsize, args.topk, args.desc_type.split(",")

    save_path = f"{args.save_path}/{dataset.lower()}"
    im_paths = read_imlist(os.path.join(save_path, args.file_name))

    if args.backbone == 'dinov2':
        global_dim = local_dim = 768
        extract_f = partial(extract_dino, im_paths=im_paths)
        model = load_dinov2()
        scale_list = [1.]
        ps = 14
    elif args.backbone == 'cvnet':
        global_dim = 2048
        local_dim = 1024
        extract_f = extract_cvnet
        model = load_cvnet(f'{_BASE_URL}/CVPR2022_CVNet_R101.pt')
        scale_list = [0.7071, 1.0, 1.4142]
        ps = None
    else:
        raise ValueError(f"Backbone {args.backbone} not supported")

    model.cuda()
    model.eval()

    detector = SpatialAttention2d(local_dim)
    detector.cuda()
    detector.eval()
    if args.pretrained:
        cpt = torch.hub.load_state_dict_from_url(f'{_BASE_URL}/{args.backbone}_detector.pt')
        detector.load_state_dict(cpt['state'], strict=True)

    if args.split == '_query' and dataset in ['roxford5k', 'rparis6k', 'instre']:
        with open(os.path.join(args.data_path, f'gnd_{dataset.lower()}.pkl'), 'rb') as fin:
            gnd = pickle.load(fin)['gnd']
        dataset = DataSet(dataset, args.data_path, scale_list, im_paths, imsize=imsize, gnd=gnd, patch_size=ps)
    else:
        dataset = DataSet(dataset, args.data_path, scale_list, im_paths, imsize=imsize, patch_size=ps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=BatchSampler(SequentialSampler(dataset), batch_size=1, drop_last=False),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    file_name = '' if file_name[-4:] == '.txt' else '_' + file_name

    feature_storage = FeatureStorage(save_path, args.backbone, args.split, file_name, global_dim, local_dim,
                                     len(dataset), desc_type, topk=topk)
    extract_f(model, detector, feature_storage, dataloader, topk)


if __name__ == "__main__":
    #  $env:PYTHONPATH = "C:\gitRepo\ames;" + $env:PYTHONPATH
    '''
    What Will Happen
    The script will now perform the following actions:

    -Load Model: It will load the pretrained DINOv2 model.
    -Read Image List: It will read the 20 image names from image_list.txt.
    -Process Images: For each of the 20 .tif images, it will:

    Load the image from SmallDb.
    Extract the top 600 local features (patches).
    Save Features: It will save all the extracted features into a single HDF5 file.
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


    ''' 

    #pathData = r'C:\OrthoPhoto\SmallDb'




    '''
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


    main()

   