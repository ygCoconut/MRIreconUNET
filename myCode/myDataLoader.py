# import matplotlib.pyplot as plt
# import scipy.io as spio
# import os
# import time
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
# from fastai.vision import *

# class MultiChannelImageImageList(ImageImageList):

"""
class MultiChannelImageImageList(ImageImageList):
    def open(self, fn):
        self.input_file_list  = glob.glob(fn + "*RADIAL*")
        self.target_file_list = glob.glob(fn + "*CARTESIAN*")

        X = np.load(self.input_file_list[idx])
        Y = np.load(self.target_file_list[idx])
        return  X['arr_0'], Y['arr_0']
"""



# # C.f. the instructions from the pytorch "DATA LOADING AND PROCESSING TUTORIAL" notebook
class CMRIreconDataset(Dataset):
    """CMRIrecon dataset."""
    def __init__(self, input_file_path, target_file_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = np.load(input_file_path)
        self.targets = np.load(target_file_path)

    def __len__(self):
#         print("print length of inputs",len(self.inputs))
#         print("print shape of inputs",np.shape(self.inputs))
        return len(self.inputs)

    def __getitem__(self, idx):

#         sample = {'input': self.inputs[idx], 'target': self.targets[idx]}
        X = self.inputs[idx]
        Y = self.targets[idx]

        return  X, Y    # np.array to torch.float32# # C.f. the instructions from the pytorch "DATA LOADING AND PROCESSING TUTORIAL" notebook


class CMRIreconDataset_withtfm_and_samplesdict(Dataset):
    """CMRIrecon dataset."""
    def __init__(self, input_file_path, target_file_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = np.load(input_file_path)
        self.targets = np.load(target_file_path)
        self.transform = transform

    def __len__(self):
#         print("print length of inputs",len(self.inputs))
#         print("print shape of inputs",np.shape(self.inputs))
        return len(self.inputs)

    def __getitem__(self, idx):

#         sample = {'input': self.inputs[idx], 'target': self.targets[idx]}
        X = self.inputs[idx]
        Y = self.targets[idx]
        sample_slice = {'input_slice': X, 'target_slice': Y}

        if self.transform is not None:
            sample_slice = self.transform(sample_slice)

        return  sample_slice    # np.array to torch.float32


class CMRIreconDataset_npzfiles(Dataset):
    """CMRIrecon dataset."""
    def __init__(self, root_dir):#, transforms = transforms.ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.input_file_list  = glob.glob(root_dir + "*RADIAL*")
        self.target_file_list = glob.glob(root_dir + "*CARTESIAN*")
        # self.c = 20
        # self.trfm = transforms

    def __len__(self):
#         print("print length of inputs",len(self.inputs))
#         print("print shape of inputs",np.shape(self.inputs))
        # return len(self.input_file_list)
        print(len(self.input_file_list))

    def __getitem__(self, idx):

        X = np.load(self.input_file_list[idx])
        Y = np.load(self.target_file_list[idx])
        # return  torch.Tensor(X['arr_0']), torch.Tensor(Y['arr_0'])
        return  X['arr_0'], Y['arr_0']
        # return  np.array(X['arr_0']), np.array(Y['arr_0'])
