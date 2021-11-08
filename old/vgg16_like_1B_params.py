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

            nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=3),
                nn.BatchNorm3d(64),
                nn.ReLU(),
            ),

            # nn.Sequential(
            #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=3),
            #     nn.BatchNorm3d(64),
            #     nn.ReLU(),
            # ),

            nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=3),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            ),

            nn.MaxPool3d(kernel_size=2,stride=2),
        )

        # --------------------------------------
        
        self.conv2 = nn.Sequential(

            nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=3),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            ),

            # nn.Sequential(
            #     nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=3),
            #     nn.BatchNorm3d(128),
            #     nn.ReLU(),
            # ),

            # nn.Sequential(
            #     nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=3),
            #     nn.BatchNorm3d(128),
            #     nn.ReLU(),
            # ),

            nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=3),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            ),

            nn.MaxPool3d(kernel_size=2,stride=2),
        )

        # --------------------------------------
        
        self.conv3 = nn.Sequential(

            nn.Sequential(
                nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=3),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            ),

            # nn.Sequential(
            #     nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=3),
            #     nn.BatchNorm3d(256),
            #     nn.ReLU(),
            # ),

            # nn.Sequential(
            #     nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=3),
            #     nn.BatchNorm3d(256),
            #     nn.ReLU(),
            # ),

            nn.Sequential(
                nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=3),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            ),

            nn.MaxPool3d(kernel_size=2,stride=2),
        )

        # --------------------------------------
        
        # self.conv4 = nn.Sequential(

        #     nn.Sequential(
        #         nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=3),
        #         nn.BatchNorm3d(512),
        #         nn.ReLU(),
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=3),
        #         nn.BatchNorm3d(512),
        #         nn.ReLU(),
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=3),
        #         nn.BatchNorm3d(512),
        #         nn.ReLU(),
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=3),
        #         nn.BatchNorm3d(512),
        #         nn.ReLU(),
        #     ),

        #     nn.MaxPool3d(kernel_size=2,stride=2),
        # )

        # --------------------------------------
        
        # 6 = 4 Conv3D layers + maxpool(2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*15*23*23, out_features=512),
            # nn.Linear(in_features=512, out_features=512),
            # nn.Linear(in_features=512*18*22*22, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(in_features=1024, out_features=1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        # )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        
        self.fc4 = nn.Linear(in_features=256, out_features=self.nbr_classes)

    def forward(self, x):

        # print("shape")
        # print(x.shape)

        print("in conv1")
        print(x.shape)
        x = self.conv1(x)

        print("in conv2")
        print(x.shape)
        x = self.conv2(x)

        print("in conv3")
        print(x.shape)
        x = self.conv3(x)

        # print("in conv4")
        # print(x.shape)
        # x = self.conv4(x)

        print("in flatten")
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print("out flatten")
        print(x.shape)

        x = self.fc1(x)
        print("fc1")
        print(x.shape)

        # x = self.fc2(x)
        # print("fc2")
        # print(x.shape)
        # print(x)

        x = self.fc3(x)
        print("fc3")
        print(x.shape)
        print(x)

        x = self.fc4(x)
        print("fc4")
        print(x.shape)
        print(x)
        
        return x
