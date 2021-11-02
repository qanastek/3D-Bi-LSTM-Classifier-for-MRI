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
normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path))]
# normal = [(getImagesFromNii(os.path.join(ct0_path, x)), 0) for x in tqdm(os.listdir(ct0_path)[0:2])]
print("Load normal: ", len(normal))
print("Load normal: ", str(type(normal[0][0])))
print("Load normal: ", normal[0][0].shape)

# abnormal = [(getImagesFromNii(os.path.join(ct23_path, x)), 1) for x in tqdm(os.listdir(ct23_path))]
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

        # print("shape")
        # print(x.shape)

        x = self.pool(F.relu(self.conv1(x)))
        # print("conv1")
        # print(x.shape)

        x = self.pool(F.relu(self.conv2(x)))
        # print("conv2")
        # print(x.shape)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print("flatten")
        # print(x.shape)

        x = F.relu(self.fc1(x))
        # print("fc1")
        # print(x.shape)

        x = F.relu(self.fc2(x))
        # print("fc2")
        # print(x.shape)

        x = self.fc3(x)
        # print("fc3")
        # print(x.shape)
        
        return x

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

model = Model3D(nbr_classes)
# model = model.float()
print(model)

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in tqdm(range(10)):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(train_loader):

        # get the inputs; data is a list of [inputs, labels]
        print("Batch ", i, "/", len(train_loader))
        inputs, labels = data
        # print(labels)

        # inputs = inputs.float()
        # labels = labels.float()

        # print(labels)
        # print(type(labels))
        # print(type(inputs))

        # inputs = Variable(inputs.view(inputs.size(0), -1))
        # inputs = Variable(inputs.view(BATCH_SIZE,LAYERS_DEEPTH,WIDTH,HEIGHT))
        inputs = Variable(inputs.view(BATCH_SIZE,CHANNELS,LAYERS_DEEPTH,WIDTH,HEIGHT))
        # print(inputs.shape)
        # inputs = Variable(inputs)
        labels = Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # print(inputs)
        # print(inputs.shape)
        # print(type(inputs))

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = "./model3d.pth"
torch.save(model.state_dict(), PATH)

dataiter = iter(test_loader)
images, labels = dataiter.next()

print("-"*50)
print(len(images))
print(images[0].shape)
print(type(images[0]))
print(type(images))
print("-"*25)
print(labels)
print(type(labels))
print("-"*50)

# Need to works with 3D images
# def imshow(img):
#     # img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Model3D(nbr_classes)
net.load_state_dict(torch.load(PATH))
outputs = net(images[None, ...])
_, predicted = torch.max(outputs, 1)
print("="*50)
print(test_labels.tolist())
print(predicted.tolist())
print("="*50)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data    
        outputs = net(images)    
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

  
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                   accuracy))

# for epoch in range(1):

#     for i, (images, labels) in enumerate(train_loader):
        
#         train = Variable(images.view(100,3,512,512,512))
#         labels = Variable(labels)

#         optimizer.zero_grad()
#         outputs = model(train)
#         loss = error(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         count += 1
#         if count % 50 == 0:

#             correct = 0
#             total = 0

#             # Iterate through test dataset
#             for images, labels in test_loader:
                
#                 test = Variable(images.view(100,3,512,512,512))
#                 # Forward propagation
#                 outputs = model(test)

#                 # Get predictions from the maximum value
#                 predicted = torch.max(outputs.data, 1)[1]
                
#                 # Total number of labels
#                 total += len(labels)
#                 correct += (predicted == labels).sum()
            
#             accuracy = 100 * correct / float(total)
            
#             # store loss and iteration
#             loss_list.append(loss.data)
#             iteration_list.append(count)
#             accuracy_list.append(accuracy)

#         if count % 500 == 0:
#             # Print Loss
#             print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

