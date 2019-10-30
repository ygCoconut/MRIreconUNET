from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from models import UNet2d_assembled
from myCode import myDataLoader
from myCode import LR_scheduler

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchsummary import summary

import sys
import os
import platform
import time

# from pynvml import * #GPU print infos


#######################################################################
# INITIALIZE
#######################################################################
tick = time.time() # stopwatch for runtime uses tick and tock


OS = platform.system() # OS = 'Linux' or 'Windows'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #device
NUM_GPUs = torch.cuda.device_count() # number of GPUs available


# These argvs are extracted from config.txt by master.py and passed to runall.py
# c.f. to master.py for a detailed description.
EPOCHS = int(sys.argv[1])
# TODO: # one cycle policy / scheduling for the learning rate
LR = float(sys.argv[2])
WEIGHT_DECAY = float(sys.argv[3]) # regularization strength a.k.a. weight decay
BS = int(sys.argv[4])
# TODO: #save the config file to copy it to MODELDIR
CONFIG = sys.argv[5] #save the config file to copy it to MODELDIR
# MODELDIRKEYWORD = sys.argv[5]

# MODELDIR is where all the models will be saved.
curr_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
MODELDIR = str("results/AllRuns/" + curr_time + "_E" + str(EPOCHS) + "_LR" + str(LR) +
                        "_WD" + str(WEIGHT_DECAY) + "/")
# The tensorboard logs are saved under results/RunAll/LOGDIR
# At the end of this code the logs are moved into MODELDIR
LOGDIR = str(curr_time + "_E" + str(EPOCHS) + "_LR" + str(LR) +
                        "_WD" + str(WEIGHT_DECAY))
COMMENT = LOGDIR
DATASPLIT = False
NUM_WORKERS = (0 if OS == 'Windows' else 8)

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
    # "\n TB logs saved at:\t", LOGDIR,

    )

#######################################################################
# DATASET
#######################################################################
print("\n\n############################################################")
print("\n\t\tDATASET\n")
print("############################################################\n")
print("\nLoading pytorch dataset...")

if DATASPLIT == True:
    if OS == 'Linux':
        CMRIdataset = myDataLoader.CMRIreconDataset(
            input_file_path = '/home/nw92/reconproject_data/input_data.npy',
            target_file_path = '/home/nw92/reconproject_data/target_data.npy')
    elif OS == 'Windows':
        CMRIdataset = myDataLoader.CMRIreconDataset(
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/input_data.npy', \
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/target_data.npy')
    else:
        print("Please use valid COMPUTER.\nOptions:\t\'Windows\'\t\'Linux\'")

    print("Split into train and valid set...")
    data_train, data_valid = train_test_split(CMRIdataset, test_size = 0.2)
### End DATASPLIT == True


if DATASPLIT == False:
    if OS == 'Linux':
        data_train = myDataLoader.CMRIreconDataset(
            '/home/nw92/reconproject_data/128x128train_data/input_data.npy', \
            '/home/nw92/reconproject_data/128x128train_data/target_data.npy')
        data_valid = myDataLoader.CMRIreconDataset(
            '/home/nw92/reconproject_data/128x128valid_data/input_data.npy', \
            '/home/nw92/reconproject_data/128x128valid_data/target_data.npy')
    elif OS == 'Windows':
        data_train = myDataLoader.CMRIreconDataset(
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/128x128train_data/input_data.npy', \
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/128x128train_data/target_data.npy')
        data_valid = myDataLoader.CMRIreconDataset(
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/128x128valid_data/input_data.npy', \
            'C:/Users/littl/Documents/PythonScripts/reconproject_data/128x128valid_data/target_data.npy')
    else:
        print("Please use valid COMPUTER.\nOptions:\t\'Windows\'\t\'Linux\'")
### End DATASPLIT == False


# load train set
train_dl = DataLoader(data_train, batch_size = BS,
                    shuffle=True, num_workers=NUM_WORKERS )
# load validation set
valid_dl = DataLoader(data_valid, batch_size = BS//2, #BS//2 to fit local GPU
                    shuffle=False, num_workers=NUM_WORKERS)

print("Loading complete!\n")


#######################################################################
# TRAINING
#######################################################################

print("\n\n############################################################")
print("\n\t\tTRAINING\n")
print("############################################################\n\n")


# TODO: ADD BEST VAL_LOSS MODEL CHECKPOINTER
# TODO: ADD LOOP OVER PARAMS, CF https://www.youtube.com/watch?v=ycxulUVoNbk

net = UNet2d_assembled.UNet2D() #20 channels
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
                            lr = LR,
                            weight_decay = WEIGHT_DECAY,
                            )

# # Implementation of cyclic learning rate.
# # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
# optimizer = torch.optim.SGD(net.parameters(), lr=LR)
# step_size = 4*len(train_dl)
# factor = 6
# end_lr = LR
# clr = LR_scheduler.cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

net.to(DEVICE)

# https://towardsdatascience.com/model-summary-in-pytorch-b5a1e4b64d25
write_summary = False
if write_summary:
    summary(net, input_size=(20, 96, 96))

inputs, targets = next(iter(train_dl))
tb = SummaryWriter(MODELDIR)#, comment = COMMENT)

for epoch in range(EPOCHS):  # loop over the dataset EPOCHS times

    total_epoch_loss_train = 0.0
    total_epoch_loss_valid = 0.0
    # TRAIN DATA LOOP in every epoch
    for i, data in enumerate(train_dl, 0):
        # get the inputs and targets
        inputs, targets = data
        # print(inputs.shape, targets.shape)
        inputs = inputs.to(torch.float32).to(DEVICE)   # to float32 and to cuda device
        targets = targets.to(torch.float32).to(DEVICE) # to float32 and to cuda device

        # zero the parameter gradients
        optimizer.zero_grad()

        # main part of training:
        outputs = net(inputs) # forward pass
        train_loss = criterion(outputs, targets) # calc loss
        train_loss.backward() # backward pass calculate gradients
        # scheduler.step() # > Where the magic happens, LR scheduler update
        optimizer.step() # update gradients

        # compute total train loss over all images
        total_epoch_loss_train += train_loss.item()
    # END TRAIN DATA LOOP

    # VAL DATA LOOP
    for i, data in enumerate(valid_dl, 0):
        # get the inputs and targets
        inputs, targets = data
        # print(inputs.shape)
        # print(inputs.shape, targets.shape)
        inputs = inputs.to(torch.float32).to(DEVICE)   # to float32 and to cuda device
        targets = targets.to(torch.float32).to(DEVICE) # to float32 and to cuda device

        # forward + loss
        outputs = net(inputs)
        valid_loss = criterion(outputs, targets)
        total_epoch_loss_valid += valid_loss.item()
    # END VAL DATA LOOP


    # PRINT TRAIN AND VAL LOSS TO CMD
    if epoch == 0:
        print("\nEPOCH\t TR_LOSS\t VAL_LOSS")   # Add as column names
    # print(epoch,"\t", round(total_epoch_loss_train/len(data_train), 5),"\t",
    #         round(total_epoch_loss_valid/len(data_valid), 5)) # round loss to 10e-5
    # print(len(train_dl))
    print(epoch,"\t", round(total_epoch_loss_train/len(train_dl), 5),"\t",
            round(total_epoch_loss_valid/len(valid_dl), 5)) # round loss to 10e-5

    # ADD INSTANCES TO TENSORBOARD
    # Add loss to tensorboard
    tb.add_scalar('Train Loss', total_epoch_loss_train/len(train_dl), epoch)
    tb.add_scalar('Valid Loss', total_epoch_loss_valid/len(valid_dl), epoch)
    # tb.add_scalar('Learning Rate', scheduler.get_lr(), epoch)
    # Add gradients and weights histogram to tensorboard every 10 epochs
    if (epoch+1) % 10  == 0: # print every 10 epochs
        for name, param in net.named_parameters():
            tb.add_histogram(name, param, epoch)
            tb.add_histogram(f'{name}.grad', param.grad, epoch)
# END TRAINING LOOP

# Add graph to tensorboard at the end of the training
tb.add_graph(net, inputs.to(torch.float32).to(DEVICE))

# TENSORBOARD instructions:
# a) Run b) in the CMD LINE:
# b) tensorboard --logdir results/AllRuns
# c) Open port 6006 in a browser. ex.: http://desktop-psfe2v8:6006/
# d) Bravo, now you can visualize your tensorboard !

tb.close()
print('\nFinished Training')


if EPOCHS > 1: # don't save the test runs, save memory
    #save the model
    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                },
                MODELDIR + 'model')
    #copy the config file to the MODELDIR
    if OS == 'Linux': # save the
        output_config=str("cp " + CONFIG + " " + MODELDIR)
    else:
        output_config=str("COPY \"" + CONFIG + "\" \"" + MODELDIR + "\"")
    print(output_config)
    os.system(output_config)

tock = time.time() # stopwatch for runtime uses tick and tock
print("\n\n############################################################")
print("\n\t\tTRAINING FINISHED!\n")
print("\t\tTOTAL RUN TIME:", round(tock - tick, 2), "secs")
print("\t\t\t\t ", round( (tock - tick)/60, 2), "min")
print("\t\tPER EPOCH AVG:\t", round( (tock - tick)/EPOCHS , 2), "secs")
print("\n############################################################\n\n")
