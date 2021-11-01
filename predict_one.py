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

# Ratios
RATIO_TRAIN = 0.90
RATIO_TEST = 0.10

# Images Specs
CHANNELS = 1
WIDTH = 128
HEIGHT = 128
# WIDTH = 512
# HEIGHT = 512
LAYERS_DEEPTH = 64

# Hyper parameters
BATCH_SIZE = 3
# BATCH_SIZE = 16

def getImagesFromNii(niiPath):
        
    scan = nib.load(niiPath).get_fdata()

    w, h, nbr_layers = scan.shape

    w_ratio = 1/(w/WIDTH)
    h_ratio = 1/(h/HEIGHT)
    d_ratio = 1/(nbr_layers/LAYERS_DEEPTH)

    # Create intermediate images
    scan = ndimage.zoom(scan, (w_ratio, h_ratio, d_ratio), order=1)

    tensor_3d = np.zeros((WIDTH,HEIGHT,LAYERS_DEEPTH))

    for i in range(LAYERS_DEEPTH):
            
        layer = scan[:, :, i]

        im = Image.fromarray(layer).convert("L").rotate(90)
        im.thumbnail((WIDTH,HEIGHT), Image.ANTIALIAS)
        
        tensor_img = transforms.ToTensor()(im)
        tensor_3d[:,:,i] = tensor_img

    tensor_3d = np.transpose(tensor_3d, (2,0,1))
    return tensor_3d

all = [(file_path, 0)]

# Test Dataset
test_images = np.asarray([getImagesFromNii(d[0]) for d in all])
test_labels = np.asarray([d[1] for d in all])
test_images, test_labels = torch.from_numpy(test_images).float(), torch.from_numpy(test_labels).long()
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("*"*50)
print("Size test_images: ", len(test_images))
print("*"*50)

classes = ('normal','abnormal')
nbr_classes = len(classes)

class Model3D(nn.Module):
    def __init__(self,nbr_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nbr_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = "./model3d.pth"
net = Model3D(nbr_classes)
net.load_state_dict(torch.load(PATH))

predictions = []

print(iter(test_loader))
print(len(iter(test_loader)))

dataiter = iter(test_loader)

for index in range(len(iter(test_loader))): 
    images, labels = next(dataiter)
    outputs = net(images[:, None, :, :, :])
    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.tolist())

print("="*50)
print([classes[a] for a in predictions])
print("="*50)
