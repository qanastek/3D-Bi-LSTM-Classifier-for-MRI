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

# Images Specs
CHANNELS = 1
WIDTH = 256
HEIGHT = 256
LAYERS_DEEPTH = 64

def getGifFromNii(niiPath, output_name):

    scan = nib.load(niiPath).get_fdata()

    w, h, nbr_layers = scan.shape

    w_ratio = 1/(w/WIDTH)
    h_ratio = 1/(h/HEIGHT)
    d_ratio = 1/(nbr_layers/LAYERS_DEEPTH)

    # RAW
    gif = []
    for i in range(nbr_layers):
        layer = scan[:, :, i]
        im = Image.fromarray(layer).convert("L")
        gif.append(im)

    gif[0].save("dev_gifs/" + output_name + "_raw" + ".gif", save_all=True, append_images=gif, loop=0)

    scan = ndimage.rotate(scan, random.choice([-20, -10, -5, 5, 10, 20]), reshape=False)
    scan = ndimage.zoom(scan, (w_ratio, h_ratio, d_ratio), order=1)
    tensor_3d = np.zeros((WIDTH,HEIGHT,LAYERS_DEEPTH))

    # Normalized
    gif = []
    for i in range(LAYERS_DEEPTH):

        layer = scan[:, :, i]
        
        im = Image.fromarray(layer).convert("L").rotate(90)
        im.thumbnail((WIDTH,HEIGHT), Image.ANTIALIAS)
        tensor_img = transforms.ToTensor()(im)
        tensor_3d[:,:,i] = tensor_img
        # im.save("dev_gifs/" + output_name + "_normalized" + ".j/peg")
        gif.append(im)

    gif[0].save("dev_gifs/" + output_name + "_normalized" + ".gif", save_all=True, append_images=gif, loop=0)

ct0_img = "/mnt/d/Projects/Datasets/IMAGE/MosMedData (3D MRI Scan)/CT-0/study_0001.nii.gz"
getGifFromNii(ct0_img,"ct0_normal")

ct23_img = "/mnt/d/Projects/Datasets/IMAGE/MosMedData (3D MRI Scan)/CT-23/study_0939.nii.gz"
getGifFromNii(ct23_img,"ct23_abnormal")
