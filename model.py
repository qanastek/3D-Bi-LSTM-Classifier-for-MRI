import os
import random
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import nibabel as nib
from scipy import ndimage

class Model3D(nn.Module):
    def __init__(self, nbr_classes):
        super().__init__()

        self.nbr_classes = nbr_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64,  kernel_size=3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
        )

        # -------------- CLASSIFIER --------------
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*2*6*6, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
        )
        
        self.fc_out = nn.Linear(in_features=4096, out_features=self.nbr_classes)

    def forward(self, x):

        # print("shape")
        # print(x.shape)

        x = self.conv1(x)
        # print("conv1")
        # print(x.shape)

        x = self.conv2(x)
        # print("conv2")
        # print(x.shape)

        x = self.conv3(x)
        # print("conv3")
        # print(x.shape)

        x = self.conv4(x)
        # print("conv4")
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print("flatten")
        # print(x.shape)

        x = self.fc1(x)
        # print("fc1")
        # print(x.shape)

        x = self.fc2(x)
        # print("fc2")
        # print(x.shape)
        # print(x)

        x = self.fc_out(x)
        # print("fc_out")
        # print(x.shape)
        # print(x)
        
        return x
