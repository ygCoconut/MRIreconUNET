import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import os
import time

def imprepare(img):

    if type(img) == np.ndarray:
        # make sure data is numpy
        pass

    else:
        img = img.cpu() # put from gpu to cpu
        img = img.detach() # in case img is a torch tensor
        npimg = img.numpy()

    img = img / 2 + 0.5     # unnormalize
    img = torch.clamp(torch.from_numpy(img), min=0.0, max=1.0) #clamp outliers
    npimg = img.numpy()

    return npimg


def cudaLoader(model, train_set, val_set):
    """
    This function takes the model, the train set and the validation set as
    inputs and sends them to cuda devices if available.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = ('cpu') # comment/uncomment to use GPU
    print("device:\t", device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(dtype = torch.double) #typecasting
    train_set = train_set.to(device)
    val_set = val_set.to(device)
    model = model.to(device);
    return model, train_set, val_set



#target_dir can be generated with timestamp !!
def imsave(recon, target_dir, s, t):
    outputPath = ('C:/Users/littl/Documents/PythonScripts/reconproject/ \
            cmriRecon/data/' + target_dir)
    spio.savemat(outputPath + 'SliceatT{}'.format(t), \
            {'SliceatT{}'.format(t): recon[0,0,...].numpy()}) #dict w/ np value
    print("Result images saved at:\n", outputPath )



# Save files to matlab folder
def savetomatlab(target_dir, datatype, datatypestr, sample_number, overwrite = False): # e.g. savetomatlab(foo_dir, input)

    if overwrite == False:
        print("Set overwrite = True to save the results !")

#   print avoids to mistakenly overwrite previous results.
    if overwrite == True:
        # Check computer to save .mat files in right path
        if COMPUTER == "Dell":
            output_path = ('C:/Users/littl/Documents/PythonScripts/reconproject/cmriRecon/results/' + target_dir)
        elif COMPUTER == "O2":
            output_path = ('/home/nw92/reconproject/cmriRecon/results/' + target_dir)

        # Check if folder already exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        spio.savemat(output_path + '/' + datatypestr + '_sample_' + str(sample_number),
        {datatypestr + '_sample_' + str(sample_number)
        : datatype[sample_number,...]
        }) #dict w/ np value

        print("Result images saved at:\n", output_path )


#not written yet
def myDataLoader():
    pass

#not written yet
def myTrainer():
    pass
