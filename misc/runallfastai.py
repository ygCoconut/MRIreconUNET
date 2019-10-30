import fastai
from fastai.callbacks import *
from fastai.vision import *
from fastai.train import *
# from fastai.dataset import ArraysIndexDataset

from fastai_tensorboard_callback import *


from torch.utils.data import Dataset, DataLoader
from models import UNet2d_assembled
from myCode import myDataLoader

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sys
import os
import platform
import time

from pynvml import * #GPU print infos


#######################################################################
# INITIALIZE
#######################################################################
tick = time.time()


OS = platform.system()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #device
NUM_GPUs = torch.cuda.device_count()

"""
if OS == 'Linux':
    # Print the memory stats for the first GPU card:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    # List the available GPU devices:
    # try:
    #     deviceCount = nvmlDeviceGetCount()
    #     for i in range(deviceCount):
    #         handle = nvmlDeviceGetHandleByIndex(i)
    #         print("Device", i, ":", nvmlDeviceGetName(handle))
    # except NVMLError as error:
    #     print(error)
"""

EPOCHS = int(sys.argv[1])
LR = float(sys.argv[2]) # one cycle policy for the learning rate
WEIGHT_DECAY = float(sys.argv[3]) # regularization strength a.k.a. weight decay
BS = int(sys.argv[4])

CONFIG = sys.argv[5] #save the config file to copy it to MODELDIR
# MODELDIRKEYWORD = sys.argv[5]

# MODELDIR is where all the models will be saved.
curr_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
MODELDIR = str("results/AllRuns/" + curr_time + "_E" + str(EPOCHS) + "_LR" + str(LR) +
                        "_WD" + str(WEIGHT_DECAY) + "/")
# The tensorboard logs are saved under logs/LOGDIR
# At the end of this code the logs are moved into MODELDIR
LOGDIR = str(curr_time + "_E" + str(EPOCHS) + "_LR" + str(LR) +
                        "_WD" + str(WEIGHT_DECAY))

print("\n--------------------------------------------------")
print("Current time marker: ", curr_time)
print("System Arguments of runall:")
print("python [RUNALL, EPOCHS, LR, WEIGHT_DECAY, BS, CONFIG]")
print(sys.argv)
print("--------------------------------------------------\n")


print("n############################################################\n")
print("\t\tVARIABLES\n")
print("############################################################\n")
print(
    "\n Device:\t\t", DEVICE,
    "\n #GPUs available:\t", NUM_GPUs,
    "\n Operating Sys:\t\t", OS,
    "\n Learning rate:\t\t", LR,
    "\n Weight decay:\t\t", WEIGHT_DECAY,
    "\n Batch size:\t\t", BS,
    "\n Epochs:\t\t", EPOCHS,
    "\n Model saved at:\t", MODELDIR,
    "\n TB logs saved at:\t", LOGDIR,

    )

#######################################################################
# DATASET
#######################################################################


############# INITIALIZE PYTORCH DATASET CLASS ########################

print("\n\n############################################################")
print("\n\t\tDATASET\n")
print("############################################################\n")
print("\nLoading data...\n")

# Pytorch dataset
train_pytorch_dataset = myDataLoader.CMRIreconDataset(
'C:/Users/littl/Documents/PythonScripts/newMRIrecon/data/processed_data/96x96_cropped_data/train/')
valid_pytorch_dataset = myDataLoader.CMRIreconDataset(
'C:/Users/littl/Documents/PythonScripts/newMRIrecon/data/processed_data/96x96_cropped_data/train/')

# train_dataset = myDataLoader.CMRIreconDataset("data/processed_data/96x96_cropped_data/train")
# valid_dataset = myDataLoader.CMRIreconDataset("data/processed_data/96x96_cropped_data/valid")


# https://github.com/fastai/fastai/blob/master/fastai/data_block.py
# # Pytorch dataset --> Fastai dataset
# train_fastai_dataset = ArraysIndexDataset(train_pytorch_dataset, transform=None)
# valid_fastai_dataset = ArraysIndexDataset(valid_pytorch_dataset, transform=None)
# # valid_fastai_dataset = ArraysIndexDataset(X_train, Y_valid, transform=None)



print(type(train_pytorch_dataset))
print(type(train_pytorch_dataset))

print("Load train_dl and valid_dl...")
# load train set
train_dl = DataLoader(train_pytorch_dataset, batch_size = BS,
                    shuffle=True, num_workers=0)
# load validation set
valid_dl = DataLoader(train_pytorch_dataset, batch_size = BS,
                    shuffle=False, num_workers=0)

# print("data type:\t",type(train_dl))
# print("data type:\t",type(train_dataset))
# print("data type:\t",type(valid_dataset))
# print(" Size of training data:", train_dataset.shape())
# print(" Size of validation data:", valid_dataset.shape())
print("\nLoading complete!\n")

# train_data = ImageImageList.from_folder('')

# DEFINE DATABUNCH TO FEED THE MODEL
my_databunch = DataBunch(train_dl,
        valid_dl,
        # test_dl,
        device=DEVICE,
        # dl_tfms:Optional[Collection[Callable]]=None,
        # path:PathOrStr='.',
        # collate_fn:Callable='data_collate',
        # no_check:bool=False
        )

#data.show_batch()

print("\n\n############################################################")
print("\n\t\tTRAINING\n")
print("############################################################\n\n")

# DEFINE LEARNER
# my_loss_func = mixup(nn.MSELoss())
my_loss_func = nn.MSELoss()
my_metrics = mean_absolute_error
my_model = UNet2d_assembled.UNet2D(20) #20 channels

# csvlogger = CSVLogger
# csvlogger.filename = str('history' + LOGDIR + '.csv')
# my_callback_fns = [csvlogger,
#             # SaveModelCallback(learn, every='epoch', monitor='valid_loss'),
#             ]

learn = Learner(data = my_databunch,
        model = my_model,
        # opt_func:Callable='Adam',
        loss_func = my_loss_func,
        metrics = my_metrics,
        # callback_fns = my_callback_fns,
        # true_wd:bool=True,
        # bn_wd:bool=True,
        wd = WEIGHT_DECAY,
        # train_bn:bool=True,
        # path:str=None,
        model_dir= (MODELDIR + 'model'),
        # callback_fns:Collection[Callable]=None,
        # callbacks:Collection[Callback]=<factory>,
        # layer_groups:ModuleList=None,
        # add_time:bool=True,
        # silent:bool=None
        ) # .to_fp16()

# MixedPrecision training
# https://docs.fast.ai/callbacks.fp16.html#MixedPrecision
# learn = learn.to_fp16()

learn.summary()

# learn.lr_find() # tested in lr_finder notebook !
# https://docs.fast.ai/callbacks.one_cycle.html
# superconvergence
# max LR =!= 6e-1
csvlogger = CSVLogger(learn, str('history' + LOGDIR))

learn.fit_one_cycle(EPOCHS, LR,
            callbacks=[
                csvlogger,
                TensorboardLogger(learn, LOGDIR),
                # SaveModelCallback(learn, every='epoch', monitor='valid_loss'),
                # MixUpCallback(learn, alpha=0.4,
                # stack_x=False, stack_y=False)
                ])

# print(learn.summary)


learn.save("trained_model")

# ISSUE
# https://github.com/fastai/fastai/issues/1663
# learn.export() #export model for inference


# print(learn.model)
# print(learn.summary())
# learn.summary()
# print(model_summary(learn.model))


# Move the history file of the CSVLogger from the default folder to MODELDIR
# Move the tensorboard log file from the default folder into the MODELDIR
# Move the CONFIG file into MODELDIR
# Found no better option to do this in the callbacks
if OS == 'Linux':
    output_hist = str("mv history" + LOGDIR + ".csv " + MODELDIR )
    output_logs = str("mv logs/" + LOGDIR + " " + MODELDIR)
    output_config=str("cp " + CONFIG + " " + MODELDIR)
else:
    output_hist = str("move history" + LOGDIR + ".csv " + MODELDIR )
    output_logs = str("move logs/" + LOGDIR + " " + MODELDIR)
    output_config=str("COPY \"" + CONFIG + "\" \"" + MODELDIR + "\"")
print("\n", output_hist)
os.system(output_hist)
print(output_logs)
os.system(output_logs)
print(output_config)
os.system(output_config)


tock = time.time()
print("\n\n############################################################")
print("\n\t\tTRAINING FINISHED!\n")
print("\t\tTOTAL RUN TIME:", tock - tick)
print("\n############################################################\n\n")
