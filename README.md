# MRIreconUNET
 Reconstruction of undersampled radial cardiac MRI with a UNET

### Execute the master file to execute the code, in return it will execute the runall file.
python3 masterfile.py config.txt

The runall.py file contains the main code to run the model training. Make sure to check it out !

The config.txt file contains all the model parameters needed to train a model.

### Installs needed:
python 3.6

conda install -c anaconda scikit-learn

conda install -c anaconda pandas

pip install torchsummary

conda install -c conda-forge tensorflow

conda install -c conda-forge tensorboardx

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
