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

from model import Model3D

ct0_path = "/users/ylabrak/datasets/MosMedData (3D MRI Scan)/CT-0/"
ct23_path = "/users/ylabrak/datasets/MosMedData (3D MRI Scan)/CT-23/"
file_path = ct0_path + "study_0001.nii.gz"

# Ratios
RATIO_TRAIN = 0.90
RATIO_TEST = 0.10

# Images Specs
CHANNELS = 1
WIDTH = 128
HEIGHT = 128
# WIDTH = 256
# HEIGHT = 256
# WIDTH = 512
# HEIGHT = 512
LAYERS_DEEPTH = 64

# Hyper parameters
BATCH_SIZE = 3
# BATCH_SIZE = 16

AUGMENTATION = True

CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# print(testset[0])
# print(type(testset[0]))
# print(testset[0][0])
# print(type(testset[0][0]))
# print(testset[0][0].shape)
# exit(0)

def getImagesFromNii(niiPath):
        
    scan = nib.load(niiPath).get_fdata()

    w, h, nbr_layers = scan.shape

    w_ratio = 1/(w/WIDTH)
    h_ratio = 1/(h/HEIGHT)
    d_ratio = 1/(nbr_layers/LAYERS_DEEPTH)

    # Create intermediate images
    if AUGMENTATION == True:
        scan = ndimage.rotate(scan, random.choice([-20, -10, -5, 5, 10, 20]), reshape=False)

    scan = ndimage.zoom(scan, (w_ratio, h_ratio, d_ratio), order=1)

    tensor_3d = np.zeros((WIDTH,HEIGHT,LAYERS_DEEPTH))
    # tensor_3d = np.zeros((CHANNELS,WIDTH,HEIGHT,LAYERS_DEEPTH))
    # tensor_3d = torch.zeros((WIDTH,HEIGHT,LAYERS_DEEPTH))

    for i in range(LAYERS_DEEPTH):
            
        layer = scan[:, :, i]

        im = Image.fromarray(layer).convert("L").rotate(90)
        im.thumbnail((WIDTH,HEIGHT), Image.ANTIALIAS)
        
        tensor_img = transforms.ToTensor()(im)
        # print(tensor_img.shape)
        
        # tensor_img = tensor_img[0]
        # print(tensor_img.shape)

        # tensor_img = transforms.ToTensor()(im)[0]
        tensor_3d[:,:,i] = tensor_img
        # tensor_3d[:,:,:,i] = tensor_img

        # im.save("layers_resize/layer" + str(i) + ".jpeg")

    # print("tensor_3d: ", str(tensor_3d.shape))
    # transforms.ToPILImage()(tensor_3d[:,:,0].squeeze_(0)).save("layers_resize/ULTIMA.jpeg")

    # print(tensor_3d.shape)
    tensor_3d = np.transpose(tensor_3d, (2,0,1))
    # tensor_3d = torch.from_numpy(tensor_3d).float()
    return tensor_3d

# getImagesFromNii(file_path)
# exit(0)

normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path))]
# normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path)[0:2])]
print("Load normal: ", len(normal))
print("Load normal: ", str(type(normal[0][0])))
print("Load normal: ", normal[0][0].shape)

abnormal = [(getImagesFromNii(os.path.join(ct23_path, x)), 1) for x in tqdm(os.listdir(ct23_path))]
# abnormal = [(getImagesFromNii(os.path.join(ct23_path, x)), 1) for x in tqdm(os.listdir(ct23_path)[0:2])]
print("Load abnormal: ", len(abnormal))

all = normal + abnormal

random.seed(0)
random.shuffle(all)

# Indexes
CORPORA_SIZE = len(all)
TRAIN_INDEX = int(CORPORA_SIZE * RATIO_TRAIN)
TEST_INDEX = int(TRAIN_INDEX + CORPORA_SIZE * RATIO_TEST)

# Training Dataset
train_images = np.asarray([d[0] for d in all[:TRAIN_INDEX]])
train_labels = np.asarray([d[1] for d in all[:TRAIN_INDEX]])
train_images, train_labels = torch.from_numpy(train_images).float(), torch.from_numpy(train_labels).long()
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test Dataset
# test_images, test_labels = map(list, zip(*all[TRAIN_INDEX:]))
test_images = np.asarray([d[0] for d in all[TRAIN_INDEX:]])
test_labels = np.asarray([d[1] for d in all[TRAIN_INDEX:]])
test_images, test_labels = torch.from_numpy(test_images).float(), torch.from_numpy(test_labels).long()
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("*"*50)
print("Size train_images: ", len(train_images))
print("Size test_images: ", len(test_images))
print("*"*50)

classes = ('normal','abnormal')
nbr_classes = len(classes)

model = Model3D(nbr_classes).to(device)
# model = model.float()
print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# exit(0)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ExponentialLR(optimizer, gamma=0.96)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

EPOCHS = 100
# EPOCHS = 2

for epoch in tqdm(range(EPOCHS)):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in tqdm(enumerate(train_loader)):

        # get the inputs; data is a list of [inputs, labels]
        # print("Batch ", i, "/", len(train_loader))
        inputs, labels = data
        # print(labels)

        # inputs = inputs.float()
        # labels = labels.float()

        # print(labels)
        # print(type(labels))
        # print(type(inputs))

        # inputs = Variable(inputs.view(inputs.size(0), -1))
        # inputs = Variable(inputs.view(BATCH_SIZE,LAYERS_DEEPTH,WIDTH,HEIGHT))
        inputs = Variable(inputs.view(BATCH_SIZE,CHANNELS,LAYERS_DEEPTH,WIDTH,HEIGHT)).to(device)
        # print(inputs.shape)
        # inputs = Variable(inputs)
        labels = Variable(labels).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # print(inputs)
        # print(inputs.shape)
        # print(type(inputs))

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0: 
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    scheduler.step()

print('Finished Training')

PATH = "./models/" + str(EPOCHS) + "_model3D_" + CURRENT_DATE + ".pth"
torch.save(model.state_dict(), PATH)
