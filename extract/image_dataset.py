#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import h5py
import numpy as np

from extract.transforms import random_sized_crop, color_norm
import torch.utils.data

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


class FeatureStorage:
    def __init__(self, save_dir, desc_name, split, extension, global_desc_dim, local_desc_dim, dataset_size, save_type, topk=400):
        self.save_dir = save_dir
        self.desc_name = desc_name
        self.dataset_size = dataset_size
        self.save_type = save_type
        self.pointer = 0
        self.storage = {}

        for desc_type in save_type:
            fnameSave=os.path.normpath(os.path.join(save_dir, f'{desc_name}{split}_{desc_type}{extension}.hdf5'))
            print(f"Creating HDF5 file: {fnameSave}")
            # h5py.File(path, "r", libver="latest", swmr=True)   # For reading with SWMR multithrea
            hdf5_file = h5py.File(fnameSave, 'w')
            shape = [dataset_size, topk, local_desc_dim + 5] if desc_type == 'local' else [dataset_size, global_desc_dim]
            hdf5_file.create_dataset("features", shape=shape, dtype=np.float32)
            self.storage[desc_type] = hdf5_file

    '''
    def __del__(self):
        for hdf5_file in self.storage.values():
            hdf5_file.close()
    '''
    #micmic
    def __del__(self):
        """Safely close all HDF5 files"""
        for desc_type, hdf5_file in self.storage.items():
            try:
                if hdf5_file is not None and hdf5_file.id.valid:
                    hdf5_file.close()
            except (AttributeError, TypeError, RuntimeError):
                # Ignore errors during cleanup - file may already be closed
                pass

    def close(self):
        """Explicitly close all HDF5 files"""
        for desc_type, hdf5_file in list(self.storage.items()):
            try:
                if hdf5_file is not None and hdf5_file.id.valid:
                    hdf5_file.close()
                self.storage[desc_type] = None
            except (AttributeError, TypeError, RuntimeError):
                pass
        self.storage.clear()        

    def save(self, feats, save_type):
        if save_type in self.storage:
            # If feats on GPU, convert to CPU first
            # micmic torch tensor to numpy array conversion
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            self.storage[save_type]["features"][self.pointer:self.pointer + len(feats)] = feats

    def update_pointer(self, size):
        self.pointer += size


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, name, data_path_base, scale_list, data_path, imsize=None, patch_size=None,
                 gnd=None, train=False, norm=True):
        assert os.path.exists(
            data_path_base), "Data path '{}' not found".format(data_path_base)
        self.data_path_base = data_path_base
        self.data_path = data_path
        self.gnd = gnd
        self.name = name
        self._scale_list = np.asarray(scale_list)
        self.imsize = imsize
        self.ps = patch_size
        self.train = train
        self.norm = norm
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        if self.name == 'gldv2' and self.train:
            samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3]))
                       for
                       line in self.data_path]
            self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
            cat_to_label = dict(zip(self.categories, range(len(self.categories))))
            samples = [(entry[0], cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
            self.targets = np.asarray([entry[1] for entry in samples])

        self.data_path = [os.path.join(self.data_path_base, i.split(',')[0]) for i in self.data_path]
        self.n = len(self.data_path)

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        if self.train:
            im = random_sized_crop(im, self.imsize)
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = torch.from_numpy(im)

        if self.norm:
            im = color_norm(im, _MEAN, _SD)
        return im

    def quantization_factor(self, side, scale):
        new_side = scale * side
        quantize_to = max(round(new_side / self.ps), 1.0)
        return scale / ((new_side / self.ps) / quantize_to)

    def __getitem__(self, indices):
        # Load the image
        im_list, scale_list = [], []
        for index in indices:
            try:
                img = cv2.imread(os.path.join(self.data_path_base, self.data_path[index]))

                if not self.train and self.imsize is not None:
                    scales = self._scale_list * (self.imsize / max(img.shape))
                else:
                    scales = self._scale_list

                if self.gnd is not None:
                    bbx = self.gnd[index]["bbx"]
                    img = img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]

                for scale in scales:
                    if not self.train and self.ps is not None:
                        scale_x = self.quantization_factor(img.shape[1], scale)
                        scale_y = self.quantization_factor(img.shape[0], scale)
                    else:
                        scale_x, scale_y = scale, scale

                    if scale_x < 1.0 or scale_y < 1.0:
                        im = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
                    elif scale_x > 1.0 or scale_y > 1.0:
                        im = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img

                    im_np = im.astype(np.float32, copy=False)
                    im_list.append(im_np)
                    scale_list.append((scale_x, scale_y))

            except Exception as e:
                print(f'error: {e}, file: {self.data_path[index]}')

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])

        if self.train:
            return im_list, self.targets[indices]
        else:
            return im_list, scale_list

    def __len__(self):
        return len(self.data_path)
