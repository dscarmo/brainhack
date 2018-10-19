import os
import glob
import multiprocessing as mp
from sys import argv

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm

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
        image = torch.unsqueeze(torch.from_numpy(npimage), 1).float()
        mask = torch.unsqueeze(torch.from_numpy(npmask), 1).float()
        return image, mask

class ReconstructionDataset(data.Dataset):
    '''
    Holds methos to handle k-space/mri data
    '''
    undersample_mask = np.load("sampling_mask_25perc.npy")
    @staticmethod
    def test():
        rd = ReconstructionDataset()
        print(len(rd))
        for i in range(len(rd)):
            img, uimg = rd[i]
            cv.imshow("img", img)
            cv.imshow("uimg", uimg)
            if cv.waitKey(16) == 27:
                break

    def self_test(self):
        print("Testing RD in {} mode".format(self.mode))
        for i in range(len(self)):
            img, uimg = self.__getitem__(i)
            if type(img) is np.ndarray:
                cv.imshow("img", img)
                cv.imshow("uimg", uimg)
            else:
                cv.imshow("img", img.squeeze().numpy())
                cv.imshow("uimg", uimg.squeeze().numpy())
            if cv.waitKey(1) == 27:
                break                
    
    # def __init__(self, root=os.path.join("E:\\", "fast_bigdata", "Raw-data"), mode="Train", transform=None):
    def __init__(self, root=os.path.join("/home", "diedre", "bigdata", "compressed_data", "Raw-data"), mode="train", transform=None, load=True):
        '''
        root: Root of folder with Train/Test/Val folders
        mode: use Train, Test or Val folder
        '''
        self.mode = mode
        self.data_ids = glob.glob(os.path.join(root, mode, "*.npy"))
        self.transform = transform
        self.volume_len = np.load(self.data_ids[0]).shape
        
        if load is False:
            self.slices = np.zeros((self.volume_len[0]*len(self.data_ids), self.volume_len[1], self.volume_len[2]), dtype=np.float)
            self.uslices = np.zeros((self.volume_len[0]*len(self.data_ids), self.volume_len[1], self.volume_len[2]), dtype=np.float)
            
            print("Loading {} data".format(mode))
            i = 0
            for filename in tqdm(self.data_ids):
                data = np.load(filename) 
                image = kspace_toimg(data)
                image = view_normalize(image)

                undersampled_data = undersample_kspace(data)
                undersampled_image = kspace_toimg(undersampled_data)
                uimage = view_normalize(undersampled_image)
                
                self.slices[170*i:170*i + 170, :, :] = image[0:170, :, :]
                self.uslices[170*i:170*i + 170, :, :] = uimage[0:170, : , :]
                i += 1
            
            np.save("{}slices.npy".format(mode), self.slices)
            np.save("{}uslices.npy".format(mode), self.uslices)
        else:
            self.slices = np.load("{}slices.npy".format(mode))
            self.uslices = np.load("{}uslices.npy".format(mode))

        self.len = self.slices.shape[0]

        if self.transform is not None:
            self.slices, self.uslices = self.transform(self.slices, self.uslices)

    def __len__(self):
        return self.len

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
        return self.uslices[i], self.slices[i]


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
    #print("Pre undersample shape: {}".format(kspace.shape))
    usampled = kspace.copy()
    usampled[:, ReconstructionDataset.undersample_mask, :] = 0

    #print("Post undersample shape: {}".format(usampled.shape))
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

def mp_datasetinit(mode, data_transforms, dict, load):
    dict[mode] = ReconstructionDataset(mode=mode, transform=data_transforms[mode], load=load)


def prepare_environment(load=True, debug=False, multithread=True, batch_size=10):
    '''
    Returns dataloaders, device and dataset_sizes
    '''    
    print("Loading data... Please wait...")
    data_transforms = {'train': ToTensor(), 'validation': ToTensor(), 'test': ToTensor()}

    modes = ['train', 'validation', 'test']
    rec = None
    if WINDOWS:
        rec = {x: ReconstructionDataset(mode=x, transform=data_transforms[x], load=load) for x in modes}
        nworkers = 0
    else:
        nworkers = 0
        if multithread:
            m = mp.Manager()
            rec = m.dict()
            ps = []
            for x in modes:
                p = mp.Process(target=mp_datasetinit, args=(x, data_transforms, rec, load))
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
        else:
            rec = {x: ReconstructionDataset(mode=x, transform=data_transforms[x], load=load) for x in modes}    

    if debug:
        for x in modes:
            rec[x].self_test()

    print("Using " + str(nworkers) + " workers")
    
    rec_dataloaders = {x: data.DataLoader(rec[x], batch_size=batch_size, shuffle=True, num_workers=nworkers) for x in modes}

    rec_it = iter(rec_dataloaders['test'])  # test iterator

    dataset_sizes = {x: len(rec[x]) for x in modes}

    print("Datasets: " + str(dataset_sizes))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device: " + str(device))

    return rec_dataloaders, device, dataset_sizes

if len(argv) > 1:
    if argv[1] == "test":
        prepare_environment(load=True, debug=True, multithread=True)
    