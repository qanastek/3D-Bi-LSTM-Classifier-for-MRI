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
from sklearn.metrics import classification_report

import nibabel as nib
from scipy import ndimage

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
# LAYERS_DEEPTH = 73

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

# normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path))]
normal = [(os.path.join(ct0_path, x), 0) for x in tqdm(os.listdir(ct0_path))]
# normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path)[0:2])]
# print("Load normal: ", len(normal))
# print("Load normal: ", str(type(normal[0][0])))
# print("Load normal: ", normal[0][0].shape)

# abnormal = [(getImagesFromNii(os.path.join(ct23_path, x)), 1) for x in tqdm(os.listdir(ct23_path))]
abnormal = [(os.path.join(ct23_path, x), 1) for x in tqdm(os.listdir(ct23_path))]
# abnormal = [(getImagesFromNii(os.path.join(ct23_path, x)), 1) for x in tqdm(os.listdir(ct23_path)[0:2])]
# print("Load abnormal: ", len(abnormal))

all = normal + abnormal

random.seed(0)
random.shuffle(all)

# Indexes
CORPORA_SIZE = len(all)
TRAIN_INDEX = int(CORPORA_SIZE * RATIO_TRAIN)
TEST_INDEX = int(TRAIN_INDEX + CORPORA_SIZE * RATIO_TEST)

# # Training Dataset
# train_images = np.asarray([d[0] for d in all[:TRAIN_INDEX]])
# train_labels = np.asarray([d[1] for d in all[:TRAIN_INDEX]])
# train_images, train_labels = torch.from_numpy(train_images).float(), torch.from_numpy(train_labels).long()
# train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test Dataset
# test_images, test_labels = map(list, zip(*all[TRAIN_INDEX:]))
test_images = np.asarray([getImagesFromNii(d[0]) for d in all[TRAIN_INDEX:]])
test_labels = np.asarray([d[1] for d in all[TRAIN_INDEX:]])
test_images, test_labels = torch.from_numpy(test_images).float(), torch.from_numpy(test_labels).long()
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("*"*50)
# print("Size train_images: ", len(train_images))
print("Size test_images: ", len(test_images))
print("*"*50)

classes = ['normal','abnormal']
nbr_classes = len(classes)

class Model3D(nn.Module):
    def __init__(self, nbr_classes):
        super().__init__()

        self.nbr_classes = nbr_classes

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64,  kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(64),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(64),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(128),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(256),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*2*6*6, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
        )
        
        self.fc2 = nn.Linear(in_features=1024, out_features=self.nbr_classes)

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

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print("flatten")
        # print(x.shape)

        x = self.fc1(x)
        # print("fc1")
        # print(x.shape)

        x = self.fc2(x)
        # print("fc2")
        # print(x.shape)
        
        return x

PATH = "./100_model3D_2021-11-03-20-39-49.pth"
# PATH = "./25_model3D_2021-11-03-19-05-08.pth"
# PATH = "./100_modified_model3d.pth"
# PATH = "./100_new_model3d.pth"
# PATH = "./1_new_model3d.pth"
# PATH = "./100_01-11-21_22H10_model3d.pth"
# PATH = "./model3d.pth"
net = Model3D(nbr_classes)
net.load_state_dict(torch.load(PATH))

predictions = []

print(iter(test_loader))
print(len(iter(test_loader)))

dataiter = iter(test_loader)

for index in tqdm(range(len(iter(test_loader)))):

    images, labels = next(dataiter)

    # print("-"*50)
    # print(len(images))
    # print(images[0].shape)
    # print(type(images[0]))
    # print(type(images))
    # print("-"*25)
    # print(labels)
    # print(type(labels))
    # print("-"*50)

    # Predict one
    # outputs = net(images)
    outputs = net(images[:, None, :, :, :])
    # outputs = net(images[None, ...])
    _, predicted = torch.max(outputs, 1)

    predictions.extend(predicted.tolist())

print("="*50)
print(test_labels.tolist())
print(predictions)
print("="*50)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

f1_score = classification_report(test_labels.tolist(), predictions, target_names=classes)
print(f1_score)

# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():

#     for data in test_loader:

#         images, labels = data    
#         outputs = net(images[:, None, :, :, :])
#         _, predictions = torch.max(outputs, 1)

#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):

#             if label == prediction:
#                 correct_pred[classes[label]] += 1

#             total_pred[classes[label]] += 1

# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
