# Train with MRI Cine Data with 20 Channels
# Date: 2nd April 2019
# Contact Ivy Chan at yanchi.chan@tum.de


import os
import time
import numpy as np
import keras
from keras import backend as K
import pylab
import matplotlib.cm as cm
import keras.layers
import tensorflow as tf

from keras.models import Sequential, model_from_json, Model
from keras.layers import Input, Dense, Dropout, Flatten, RepeatVector, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation, BatchNormalization, Lambda, concatenate, multiply, add, subtract
from keras.optimizers import SGD, Nadam, Adam
from keras.utils import multi_gpu_model
from keras.activations import softmax
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import functools
import itertools
from itertools import product
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from scipy import ndimage
from keras import losses
from scipy.io import loadmat
import tables
import scipy
import scipy.io as spio
import matplotlib.pyplot as plt
import h5py
import math
from keras.models import load_model

# Get training input and groundtruth
saveDir='/home/yyc5/MRI_CNN_Result/'
TrainDir='/home/yyc5/MRI_CNN_Result/Data/Raw_CS_permute'
def get_files(dirname, reverse=False):
    """ Return list of file paths in directory """
    # Get list of files
    filepaths = []
    for basename in sorted(os.listdir(dirname)):
        filename = os.path.join(dirname, basename)
        print(filename)
        if os.path.isfile(filename):
            filepaths.append(filename)
    return filepaths

filepaths=get_files(TrainDir)
print(filepaths)

No_mat = len(filepaths)

magCart=dict()
magRad=dict()
for i in range(No_mat):
    print(filepaths[i])
    TrainDatafile=loadmat(filepaths[i])
    magCart[i] = TrainDatafile['img_Cartesian_acq_cropped_norm_cropped']
    magRad[i] = TrainDatafile['recon_cs_12_slices_cropped_norm_cropped']
    print(magCart[i].shape)
    print(magRad[i].shape)

VadDir=r'/home/yyc5/MRI_CNN_Result/Data/Raw_CS_Validate'
filepaths=get_files(VadDir)
print(filepaths)

No_mat_V = len(filepaths)

magRadV=dict()
magCartV=dict()
for i in range(No_mat_V):
    print(filepaths[i])
    ValidateData=loadmat(filepaths[i])
    magCartV[i] = ValidateData['img_Cartesian_acq_cropped_norm_cropped']
    magRadV[i] = ValidateData['recon_cs_12_slices_cropped_norm_cropped']
    print(magCartV[i].shape)
    print(magRadV[i].shape)


# Concatenate the data in the dimensions of [no. of image, x, y, phases]
magRad_new_all = np.concatenate([magRad[i] for i in range(No_mat)],axis=0)
magCart_new_all = np.concatenate([magCart[i] for i in range(No_mat)],axis=0)

print(magRad_new_all.shape)
print(magCart_new_all.shape)
Y1_test=magCartV
X1_test=magRadV
segmentedY=magCart_new_all
imagesX=magRad_new_all
Y1 =segmentedY
X1 =imagesX
IMAGE_SIZE = (imagesX.shape[0], imagesX.shape[1], imagesX.shape[2], imagesX.shape[3])


#Load Minibatches
class DataLoader(object):

    def __init__(self, X1, Y1, IMAGE_SIZE, crop_size):
        self._idx = 0
        self.images = X1
        self.labels = Y1
        self.IMAGE_SIZE = IMAGE_SIZE
        self.num = self.IMAGE_SIZE[0]
        self.crop_size = crop_size

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.IMAGE_SIZE[1], self.IMAGE_SIZE[2], self.IMAGE_SIZE[3]))
        labels_batch = np.zeros((batch_size, self.IMAGE_SIZE[1], self.IMAGE_SIZE[2], self.IMAGE_SIZE[3]))

        x_batch = np.zeros((batch_size, crop_size, crop_size, self.IMAGE_SIZE[3]))
        y_batch = np.zeros((batch_size, crop_size, crop_size, self.IMAGE_SIZE[3]))

        for i in range(batch_size):
            image = self.images[self._idx, ...]
            label = self.labels[self._idx, ...]
            #distIm = self.distTransIm[self._idx, ...]
            images_batch[i, ...] = image
            labels_batch[i, ...] = label
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                # permutation
                perm = np.random.permutation(self.num)
                self.images[:, ...] = self.images[perm, ...]
                self.labels[:, ...] = self.labels[perm, ...]

        for i in range(batch_size):
            offset_h = 0
            offset_w = 0

            image_temp = images_batch[i, ...]
            label_temp = labels_batch[i, ...]

            x_batch[i, ...] = image_temp
            y_batch[i, ...] = label_temp

        return x_batch, y_batch, images_batch, labels_batch

def postConv(layer, act):
    layer = BatchNormalization()(layer)
    layer = Activation(act)(layer)
    return layer

# Build U-Net model
crop_size = 96
activation = 'relu'
numFilters = 32

input_img = Input(shape=(06, 96, 20))

c1 = Conv2D(numFilters, (3, 3), kernel_initializer='he_normal', padding='same') (input_img)
c1 = postConv(c1, activation)
#c1 = Dropout(0.1) (c1)
c1 = Conv2D(numFilters, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
c1 = postConv(c1, activation)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(numFilters*2, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
c2 = postConv(c2, activation)
#c2 = Dropout(0.1) (c2)
c2 = Conv2D(numFilters*2, (3, 3), kernel_initializer='he_normal', padding='same') (c2)
c2 = postConv(c2, activation)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(numFilters*4, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
c3 = postConv(c3, activation)
#c3 = Dropout(0.2) (c3)
c3 = Conv2D(numFilters*4, (3, 3), kernel_initializer='he_normal', padding='same') (c3)
c3 = postConv(c3, activation)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(numFilters*8, (3, 3), kernel_initializer='he_normal', padding='same') (p3)
c4 = postConv(c4, activation)
#c4 = Dropout(0.2) (c4)
c4 = Conv2D(numFilters*8, (3, 3), kernel_initializer='he_normal', padding='same') (c4)
c4 = postConv(c4, activation)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(numFilters*16, (3, 3), kernel_initializer='he_normal', padding='same') (p4)
c5 = postConv(c5, activation)
#c5 = Dropout(0.3) (c5)
c5 = Conv2D(numFilters*16, (3, 3), kernel_initializer='he_normal', padding='same') (c5)
c5 = postConv(c5, activation)

u6 = Conv2DTranspose(numFilters*8, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = postConv(u6, activation)
u6 = concatenate([u6, c4])
c6 = Conv2D(numFilters*8, (3, 3), kernel_initializer='he_normal', padding='same') (u6)
c6 = postConv(c6, activation)
#c6 = Dropout(0.2) (c6)
c6 = Conv2D(numFilters*8, (3, 3), kernel_initializer='he_normal', padding='same') (c6)
c6 = postConv(c6, activation)

u7 = Conv2DTranspose(numFilters*4, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = postConv(u7, activation)
u7 = concatenate([u7, c3])
c7 = Conv2D(numFilters*4, (3, 3), kernel_initializer='he_normal', padding='same') (u7)
c7 = postConv(c7, activation)
#c7 = Dropout(0.2) (c7)
c7 = Conv2D(numFilters*4, (3, 3), kernel_initializer='he_normal', padding='same') (c7)
c7 = postConv(c7, activation)

u8 = Conv2DTranspose(numFilters*2, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = postConv(u8, activation)
u8 = concatenate([u8, c2])
c8 = Conv2D(numFilters*2, (3, 3), kernel_initializer='he_normal', padding='same') (u8)
c8 = postConv(c8, activation)
#c8 = Dropout(0.1) (c8)
c8 = Conv2D(numFilters*2, (3, 3), kernel_initializer='he_normal', padding='same') (c8)
c8 = postConv(c8, activation)

u9 = Conv2DTranspose(numFilters, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = postConv(u9, activation)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(numFilters, (3, 3), kernel_initializer='he_normal', padding='same') (u9)
c9 = postConv(c9, activation)
#c9 = Dropout(0.1) (c9)
c9 = Conv2D(numFilters, (3, 3), kernel_initializer='he_normal', padding='same') (c9)
c9 = postConv(c9, activation)

outputs = Conv2D(20, (1, 1)) (c9)
predictions = postConv(outputs, 'linear')

#to use multiGPU support (comment all this out if not using GPUs)
numGPUs = 1
if numGPUs == 1:
    print("[INFO] training with 1 GPU...")
    # make model
    model = Model(inputs=input_img, outputs=predictions)
elif numGPUs > 1:
    print("[INFO] training with {} GPUs...".format(numGPUs))
    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        # initialize the model
        model = Model(inputs=input_img, outputs=predictions)
    # make the model parallel
    model = multi_gpu_model(model, gpus=numGPUs)
else:
    with tf.device("/cpu:0"):
        # initialize the model
        model = Model(inputs=input_img, outputs=predictions)
        print('Model Made!')

model = Model(inputs=input_img, outputs=predictions)
print('Model Made!')
lossMSE = losses.mean_squared_error
#lossMAE = losses.mean_absolute_error
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#compile model
print('Compiling!')
model.compile(loss=lossMSE, optimizer=nadam)

#parameters
epochNum= 150
batchSize = 32
if numGPUs <= 1:
    batchSizeFinal = batchSize
else:
    batchSizeFinal = round((numGPUs-.5)*batchSize)

#get final prediction size in case need to reshape segmentation
predictions_dim = predictions._keras_shape

# use DataLoader class to randomly flip and crop images
loader_train = DataLoader(X1, Y1, IMAGE_SIZE, crop_size)
loader_test = DataLoader(X1_test, Y1_test, IMAGE_SIZE, crop_size)
runningLoss_total=np.zeros(shape=(epochNum,2))
# train model
for e in range(epochNum):
    print('Epoch', e)
    start = time.time()
    batches = 0
    progbar = generic_utils.Progbar(math.ceil(IMAGE_SIZE[0] / batchSizeFinal) * batchSizeFinal)
    runningLoss = 0.0
    runningLossTest = 0.0
    while (batches <= IMAGE_SIZE[0] / batchSizeFinal):
        x_batch, y_batch, temp_images, temp_labels = loader_train.next_batch(batchSizeFinal)

        y_batch_crop = y_batch[:, 0:predictions_dim[1], 0:predictions_dim[2], :]

        model_loss = model.train_on_batch(x_batch, y_batch_crop)
        batches += 1
        runningLoss = ((runningLoss * (batches - 1)) + model_loss) / (batches)

        #x_batch_test, y_batch_test, temp_images, temp_labels = loader_test.next_batch(batchSizeFinal)
        model_loss_test = model.test_on_batch(X1_test[0], Y1_test[0])
        runningLossTest = ((runningLossTest * (batches - 1)) + model_loss_test) / (batches)
        progbar.add(batchSizeFinal, values=[("train_loss", runningLoss), ("val_loss", runningLossTest)])

    stop = time.time()
    duration = stop-start
    print(duration)
    runningLoss_total[e,0]=runningLoss
    runningLoss_total[e,1]=runningLossTest

    if e==4:
        model.save(saveDir+'model_4epoch_CS_011.h5')  #save model in GPU
    elif e==10:
        model.save(saveDir+'model_10epoch_CS_011.h5')  #save model in GPU)
    elif e==20:
        model.save(saveDir+'model_20epoch_CS_011.h5')  #save model in GPU)
    elif e==40:
        model.save(saveDir+'model_40epoch_CS_011.h5')  #save model in GPU)
    elif e==60:
        model.save(saveDir+'model_60epoch_CS_011.h5')  #save model in GPU)
    elif e==80:
        model.save(saveDir+'model_80epoch_CS_011.h5')
    elif e==100:
        model.save(saveDir+'model_100epoch_CS_011.h5')
    elif e==150:
        model.save(saveDir+'model_150epoch_CS_011.h5')
    else:
        model.save(saveDir+'model_epoch_CS_011.h5')  #save model in GPU
        spio.savemat(saveDir+'loss_CS_011_epoch',{"loss_CS_011_epoch":runningLoss_total})
        print(e)

# use DataLoader class to randomly flip and crop images
loader_train_final = DataLoader(X1, Y1, IMAGE_SIZE, cropy_size)
x_batch, y_batch, temp_images, temp_labels = loader_train_final.next_batch(IMAGE_SIZE[0])
predictY_temp = model.predict(x_batch, batch_size=batchSizeFinal)
score = model.evaluate(x_batch, y_batch, batch_size=batchSizeFinal)

#save model to disk
#model.save(saveDir+'model_done_CS_002.h5')  #save model in GPU
#spio.savemat(saveDir+'Loss_CS_002',{"runningloss_T_V":runningLoss_total})
#spio.savemat(saveDir+'Radial_CS_002',{"x_batch":x_batch})
#spio.savemat(saveDir+'Prediction_CS_002',{"predictY":predictY_temp })
#spio.savemat(saveDir+'Cartesian_CS_002',{"y_batch_crop":y_batch})

model.summary()
print('Yeah you may check the model now! Smile!')
