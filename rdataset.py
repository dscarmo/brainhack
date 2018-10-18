import os
import glob

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import nibabel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torchvision.utils import make_grid

data_dir = os.path.join("E:", "fast_bigdata")

class ReconstructionDataset(data.Dataset):
    '''
    Holds methos to handle k-space/mri data
    '''
    undersample_mask = np.load("sampling_mask_25perc.npy")
    print(undersample_mask.shape)
    def __init__(self, root=os.path.join("E:\\", "fast_bigdata", "Raw-data"), mode="Train"):
        '''
        root: Root of folder with Train/Test/Val folders
        mode: use Train, Test or Val folder
        '''
        self.data_ids = glob.glob(os.path.join(root, mode, "*.npy"))
        
    def __getitem__(self, i):
        data = np.load(self.data_ids[i]) 
        return data

def kspace_toimg(complex_kspace):
    real = complex_kspace[:, :, :, 0]
    comp = complex_kspace[:, :, :, 1]
    reconstructed = np.abs(np.fft.ifft2(real+1j*comp))
    return reconstructed

def logkspace(complex_kspace):
    real = complex_kspace[:, :, :, 0]
    comp = complex_kspace[:, :, :, 1]
    return np.log(1 + np.abs(real + 1j*comp))

def view_normalize(volume):
    return (volume - volume.min())/volume.max()

def undersample_kspace(kspace):
    print("Pre undersample shape: {}".format(kspace.shape))
    usampled = kspace.copy()
    usampled[:, ReconstructionDataset.undersample_mask, :] = 0

    print("Post undersample shape: {}".format(usampled.shape))
    return usampled

def visualize_orientation(winname, volume, undersampled_kspace=None, dim=0):
    shape = volume.shape
    nvolume = view_normalize(volume)
    nuvolume = None
    kspace = False
    if undersampled_kspace is not None:
        nuvolume = view_normalize(undersampled_kspace)
        kspace = True

    for i in range(shape[dim]):
        if dim == 0:
            cv.imshow(winname + "dim: {}".format(dim), nvolume[i, : , :])
            if kspace is True:
                cv.imshow(winname + "dim: {} undersampled".format(dim), nuvolume[i, :, :])
        elif dim == 1:
            cv.imshow(winname + "dim: {}".format(dim), nvolume[:, i , :])
            if kspace is True:
                cv.imshow(winname + "dim: {} undersampled".format(dim), nuvolume[:, i , :])
        elif dim == 2:
            cv.imshow(winname + "dim: {}".format(dim), nvolume[:, : , i])    
            if kspace is True:
                cv.imshow(winname + "dim: {} undersampled".format(dim), nuvolume[:, : , i])
        if cv.waitKey(0) == 27:
            return


complex_kspace = ReconstructionDataset()[0]
kspace_view = logkspace(complex_kspace)
print(kspace_view.shape)

undersampled_kspace = undersample_kspace(complex_kspace)
print(undersampled_kspace.shape)
ukspace_view = logkspace(undersampled_kspace)
print(ukspace_view.shape)

visualize_orientation("k space", kspace_view, ukspace_view)

mriimage = kspace_toimg(complex_kspace)
print(mriimage.shape)

visualize_orientation("Reconstructed", mriimage)
visualize_orientation("Reconstructed", mriimage, dim=1)
visualize_orientation("Reconstructed", mriimage, dim=2)
