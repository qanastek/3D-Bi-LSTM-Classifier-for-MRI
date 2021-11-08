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

        self.features = nn.Sequential(

            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.MaxPool3d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        
        self.classifier = nn.Sequential(

            nn.Linear(512*7*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, nbr_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)        
        return x
