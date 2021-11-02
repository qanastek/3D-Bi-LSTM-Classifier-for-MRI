import os
import random

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import nibabel as nib
from scipy import ndimage

ct0_path = "/mnt/d/Projects/Datasets/IMAGE/MosMedData (3D MRI Scan)/CT-0/"
ct23_path = "/mnt/d/Projects/Datasets/IMAGE/MosMedData (3D MRI Scan)/CT-23/"
file_path = ct0_path + "study_0001.nii.gz"

size_layers = []

def getImagesFromNii(niiPath):

    global MAX_LAYERS
        
    scan = nib.load(niiPath).get_fdata()

    w, h, nbr_layers = scan.shape

    size_layers.append(nbr_layers)

normal = [getImagesFromNii(os.path.join(ct0_path, x)) for x in tqdm(os.listdir(ct0_path))]
print(max(size_layers))
print(sum(size_layers) / len(size_layers))

abnormal = [getImagesFromNii(os.path.join(ct23_path, x)) for x in tqdm(os.listdir(ct23_path))]
print(max(size_layers))
print(sum(size_layers) / len(size_layers))
