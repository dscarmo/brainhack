import os
import glob

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

data_dir = os.path.join("E:", "fast_bigdata")

if os.name == "nt":
    WINDOWS = True
else:
    WINDOWS = False

class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''
    def __call__(self, npimage, npmask):
        '''
        input numpy image: H x W
        output torch image: C X H X W
        '''
        image = torch.unsqueeze(torch.from_numpy(npimage), 0).float()
        mask = torch.unsqueeze(torch.from_numpy(npmask), 0).float()
        return image, mask


class ToNumpy(object):
    '''
    Convert tensors in sample to ndarrays.
    '''
    def __call__(self, image, mask=None):
        '''
        input torch image: C X H X W
        output numpy image: H x W
        '''
        npimage = torch.squeeze(image).numpy()
        npmask = torch.squeeze(mask).numpy()
        return npimage, npmask


class ReconstructionDataset(data.Dataset):
    '''
    Holds methos to handle k-space/mri data
    '''
    undersample_mask = np.load("sampling_mask_25perc.npy")
    @staticmethod
    def test():
        complex_kspace = ReconstructionDataset().getkspace(0)
        kspace_view = logkspace(complex_kspace)
        print(kspace_view.shape)

        undersampled_kspace = undersample_kspace(complex_kspace)
        print(undersampled_kspace.shape)
        ukspace_view = logkspace(undersampled_kspace)
        print(ukspace_view.shape)

        visualize_orientation("k space", kspace_view, ukspace_view)

        mriimage = kspace_toimg(complex_kspace)
        undersampled_mriimage = kspace_toimg(undersampled_kspace)
        print(mriimage.shape)

        visualize_orientation("Reconstructed", mriimage, undersampled_mriimage)
        visualize_orientation("Reconstructed", mriimage, undersampled_mriimage, dim=1)
        visualize_orientation("Reconstructed", mriimage, undersampled_mriimage, dim=2)

    def __init__(self, root=os.path.join("E:\\", "fast_bigdata", "Raw-data"), mode="Train", transform=None):
        '''
        root: Root of folder with Train/Test/Val folders
        mode: use Train, Test or Val folder
        '''
        self.data_ids = glob.glob(os.path.join(root, mode, "*.npy"))
        self.transform = transform

    def getkspace(self, i):
        '''
        Get full kspace data at index i
        '''
        data = np.load(self.data_ids[i]) 
        return data

    def __getitem__(self, i):
        '''
        Get undersampled image and target at index i
        '''
        data = np.load(self.data_ids[i]) 
        image = kspace_toimg(data)
        image = view_normalize(image)

        undersampled_data = undersample_kspace(data)
        undersampled_image = kspace_toimg(undersampled_data)
        uimage = view_normalize(undersampled_image)

        if self.transform is not None:
            self.transform(image, uimage)

        return  uimage, image 


def kspace_toimg(complex_kspace):
    '''
    Converts complex kspace to image domain
    '''
    real = complex_kspace[:, :, :, 0]
    comp = complex_kspace[:, :, :, 1]
    reconstructed = np.abs(np.fft.ifft2(real+1j*comp))
    return reconstructed

def logkspace(complex_kspace):
    '''
    Transforms complex number for display
    '''
    real = complex_kspace[:, :, :, 0]
    comp = complex_kspace[:, :, :, 1]
    return np.log(1 + np.abs(real + 1j*comp))

def view_normalize(volume):
    '''
    Normalizes between 0 and 1
    '''
    return (volume - volume.min())/volume.max()

def undersample_kspace(kspace):
    '''
    Undersamples kspace with pre made mask
    '''
    print("Pre undersample shape: {}".format(kspace.shape))
    usampled = kspace.copy()
    usampled[:, ReconstructionDataset.undersample_mask, :] = 0

    print("Post undersample shape: {}".format(usampled.shape))
    return usampled

def visualize_orientation(winname, volume, undersampled_kspace=None, dim=0):
    '''
    Visualizes 3 planes of mri volume
    '''
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

def prepare_environment():
    '''
    Returns dataloaders, device and dataset_sizes
    '''    

    data_transforms = {'train': ToTensor(), 'Validation': ToTensor(), 'test': ToTensor()}

    modes = ['train', 'validation', 'test']

    rec = {x: ReconstructionDataset(mode=x, transform=data_transforms[x]) for x in modes}

    if WINDOWS:
        nworkers = 0
    else:
        nworkers = 12

    print("Using " + str(nworkers) + " workers")
    
    rec_dataloaders = {x: data.DataLoader(rec[x], batch_size=20, shuffle=True, num_workers=nworkers) for x in modes}

    rec_it = iter(rec_dataloaders['test'])  # test iterator

    dataset_sizes = {x: len(rec[x]) for x in modes}

    print("Datasets: " + str(dataset_sizes))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device: " + str(device))

    return rec_dataloaders, device, dataset_sizes
#ReconstructionDataset().test()