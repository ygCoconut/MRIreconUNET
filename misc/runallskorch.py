#!/usr/bin/env python
# coding: utf-8

# # This script allows you to run the model on the cluster.
# runall.py is not on an actual version right now but runall_onO2.py is
# Both scripts will be merged eventually to one.

# # Code starts here


#######################################################################
# IMPORT STATEMENTS
#######################################################################

# ACHTUNG: http://thomas-cokelaer.info/blog/2011/09/382/
# Errors when reloading module with the class in jupyter notebook !
from models import modelRepository as mr
from myCode import myFunctions
from myCode import myDataLoader
from models import UNet3d_parts
from models import UNet3d_assembled
from models import UNet2d_parts
from models import UNet2d_assembled

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


import numpy as np
import matplotlib.pyplot as plt
#import scipy
import scipy.io as spio

import os
import platform
import time



# # Setup Tensorboard
# ### Tensorboard from TF 1.0 has been used, as TF 2.0 is not fully compatible with conda yet. This will avoid possible issues when copying the conda environment over to the computer cluster O2

# import tensorflow as tf
# from tensorflow import summary
from tensorboard import notebook
from tensorboardX import SummaryWriter
from tensorboardX import FileWriter


# import skorch
from skorch import NeuralNet
import skorch.callbacks
from skorch.callbacks import Checkpoint, TrainEndCheckpoint
from skorch.callbacks import EpochScoring

# import sklearn
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# import distributed.joblib  # imported for side effects
# from sklearn.externals.joblib import parallel_backend

import torchbearer # Library for callbacks, especially TensorboardX CBs


#######################################################################
# CONSTANT VALUE INITIALIZATION
#######################################################################

#tensorboardX setup
curr_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
logdir = os.path.join("logs", curr_time)
logdir_train = os.path.join(logdir, "train")
logdir_val = os.path.join(logdir, "val")
# train_summary_writer = SummaryWriter(logdir_train) # train SummaryWriter
# val_summary_writer = SummaryWriter(logdir_val)     # validation SummaryWriter
print("\n--------------------------------------------------")
print("current time marker: ", curr_time)
print("Tensorboard log directory location:", logdir)
print("--------------------------------------------------\n")


#Constant values
EPOCHS = 1000
BS = 128 # batch size
LR = 1.5e-2
LAMBDA = 5e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #device
NUM_GPUs = torch.cuda.device_count()
# COMPUTER = "O2" #computing device

COMPUTER = "O2"
OS = platform.system()
SPLIT_TRAIN_AND_TEST = True
WARMSTART = False
SAVE = False

#print constants
print("\n\n############################################################\n")
print("\t\tVARIABLES\n")
print("############################################################\n")

print(
    "\n Epochs:\t", EPOCHS,
    "\n Regularization Strength:\t", LAMBDA,
    "\n Learning Rate:\t", LR,
    "\n Batch Size:\t", BS,
    "\n Device:\t", DEVICE,
    "\n #GPUs available:\t", NUM_GPUs,
    "\n Operating Sys:\t", OS,
    "\n Computer used:\t", COMPUTER,
    "\n Split into train and test data:\t", SPLIT_TRAIN_AND_TEST,
    "\n Warm start:\t", WARMSTART,
    "\n Save Results:\t", SAVE,

    )



#######################################################################
# DATASET
#######################################################################


############# INITIALIZE PYTORCH DATASET CLASS ########################

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
        return  X, Y

print("\n\n############################################################")
print("\n\t\tDATASET\n")
print("############################################################\n")
print("\nLoading data...\n")

if OS == 'Linux':
    CMRIdataset = CMRIreconDataset(
        input_file_path = \
        '/home/nw92/reconproject_data/input_data.npy', \
        target_file_path = \
        '/home/nw92/reconproject_data/target_data.npy')

elif OS == 'Windows':
    CMRIdataset = CMRIreconDataset(
        input_file_path = \
        'C:/Users/littl/Documents/PythonScripts/reconproject_data/input_data.npy', \
        target_file_path = \
        'C:/Users/littl/Documents/PythonScripts/reconproject_data/target_data.npy')

else:
    print("Please use valid COMPUTER.\nOptions:\t\'Windows\'\t\'Linux\'")
#    print("Note that OS has no option for Mac implemented.")
# print(CMRIdataset[:]['input'].shape)
X, Y = CMRIdataset[:][:]
print("data type:\t",type(X), X.dtype)
print("data shape:\t", X.shape)
print("\nLoading complete!\n")
print("############################################################")
print("############################################################\n")



############## Split Dataset into train set ( 80% ) ################
#################### and validation set ( 20% ) ####################

# ## This Method is possibly not ideal yet, as the two generated dataset do not
# have the same amount of slices from each heart layer.
split = SPLIT_TRAIN_AND_TEST
if split == True:
    print("\nSplit dataset into train data (80%) and test data (20%)...\n")
    # train_size = int(0.8 * len(CMRIdataset))
    # test_size = len(CMRIdataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(CMRIdataset, [train_size, test_size])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)


elif split == False:
    # X and Y are passed on to stay compatible
    X_train = X
    Y_train = Y

print("Splitting complete!\nTraining data infos:\n")
print("Train data type:\t", type(X_train), X_train.dtype)
print("Train data shape:\t", X_train.shape)
print("Test data type:\t\t", type(X_test), X_test.dtype)
print("Test data shape:\t", X_test.shape)
print("\n")



# define loaders
# ! numworkers set to 0 for windows !!
loader = False
if loader == True:
    print("load trainloader and valloader")
    # load train set
    trainloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    # load validation set
    valloader = DataLoader(val_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    print("trainloader and valloader loaded!")

    # X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size = 0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)


# # Note about NN input.
# ## c.f. NN tutorial from pytorch
#
# ``torch.nn`` only supports mini-batches. The entire ``torch.nn`` package only supports inputs that are a mini-batch of samples, and not a single sample.
#
# For example, ``nn.Conv2d`` will take in a 4D Tensor of
# ``nSamples x nChannels x Height x Width``.
#
# If you have a single sample, just use ``input.unsqueeze(0)`` to add
# a fake batch dimension.</p></div>



#######################################################################
# INCLUDE code for multiple GPUs !! #
#######################################################################
#Not working yet !
# if NUM_GPUs < 1:
#     dask-scheduler
#     for i in range(NUM_GPUs):
#         CUDA_VISIBLE_DEVICES=i dask-worker 127.0.0.1:8786 --nthreads 1
#         CUDA_VISIBLE_DEVICES=i dask-worker 127.0.0.1:8786 --nthreads 1



#######################################################################
# CALLBACK FUNCTIONS
#######################################################################

################## skorch_callbacks ###################################

# checkpoint saver
# monitor = lambda net: all(net.history[-1, (
# 	'train_loss_best', 'valid_loss_best')])
cp_best_model = Checkpoint(
    monitor = 'valid_loss_best',
    # monitor = monitor,
    dirname='results/model_checkpoints/best_model',
    )

cp_best_train = Checkpoint(
    monitor = 'train_loss_best',
    # monitor = monitor,
    dirname='results/model_checkpoints/best_train',
    )

# learning rate scheduler
cyclicLR = skorch.callbacks.LRScheduler(
            policy = 'CyclicLR',
            )
# lr = skorch.callbacks.CyclicLR(optimizer = Adam)
# lr_scheduler = LRScheduler(policy="StepLR",    step_size=7,    gamma=0.1)

#display progressbar == True
progressbar = skorch.callbacks.ProgressBar()

# for scoring check the "Scoring" section of
# https://skorch.readthedocs.io/en/stable/user/callbacks.html
def mean_abs_error_train(y_true, y_pred):
    #computes MAE. only supports 2D and 1D input --> flatten
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())
MAE_scorer = make_scorer(mean_abs_error_train, greater_is_better=True)
epoch_MAE_train = EpochScoring(
    MAE_scorer,
    name = 'MAE_scorer',
    # lower_is_better = True
)


######################################################################
# DEFINE MODEL
######################################################################

# ## Import model from modelRepository.py
net = mr.BNv0()
net2 = mr.BNv1()
channels = 20
model = UNet3d_assembled.UNet3d(channels)
model = net2

# override with actual model choice
model = mr.BN20channels()
model = UNet2d_assembled.UNet2D(channels)
# print(type(model))

model = NeuralNet(module = model,
                criterion = nn.MSELoss,
                max_epochs = EPOCHS,
                batch_size = BS,
                iterator_train__shuffle = True,
                lr = LR,
                optimizer__weight_decay = LAMBDA,
                device = DEVICE,
                callbacks = [
                cp_best_model,
                cp_best_train,
                progressbar,
                # cyclicLR,
                epoch_MAE_train,
                ]

                )


#######################################################################
# TRAIN MODEL
#######################################################################
print("############################################################")
print("\n\t\tTRAINING MODEL\n")
print("############################################################\n")


model.initialize()
# print("size of inputs fed to optimizer:\t", len(X_train), len(Y_train))
model.fit(X_train, Y_train)


# KFold cross-validation Gridsearch
# from sklearn.model_selection import GridSearchCV
# params = {
#     'optimizer__weight_decay': [0.0005, 0.001, 0.0005, 0.001, 0.005, 0.01],
#     'max_epochs': [40],
#     'lr': [0.05]
# }
# gs = GridSearchCV(model, params, refit=False, scoring = MAE_scorer)
#
# gs.fit(X_train, Y_train)
# print("Best grid search score:\t", gs.best_score_, gs.best_params_)
