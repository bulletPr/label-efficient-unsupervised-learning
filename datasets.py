'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py
from glob import glob
import torch

dataset_path = 'modelnet40_ply_hdf5_2048'

def load_h5(path):
    all_data = []
    all_label = []
    for h5_name in path:
        f = h5py.File(h5_name)
        data = f['data'][:]
        label = f['label'][:]
        all_data.append(data)
        all_label.append(label)
    return (all_data, all_label)


class ModelNetH5Dataset(object):
    def __init__(self, root=dataset_path, npoints=2048, train=False):
        self.root=root
        self.npoints = npoints
        self.h5_files = []
        if(train):
            list_filename = os.path.join(self.root, '*train*.h5')
        else:
            list_filename = os.path.join(self.root, '*train*.h5')

        self.h5_files += glob(list_filename)
        data, labels = load_h5(self.h5_files)
        self.data = np.concatenate(data,axis=0)
        self.labels = np.concatenate(labels, axis=0)

    def __getitem__(self, index):
        point_set = self.data[index]
        label = self.labels[index]
        #print(point_set.shape, seg.shape)

        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set, label


    def __len__(self):
        return self.data.shape[0]