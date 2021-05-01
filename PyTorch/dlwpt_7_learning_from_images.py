import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.

matplotlib.use('Agg')
data_path = '../DL_PyTorch/data-cifar10/'
cifar10 = datasets.CIFAR10(data_path, train = True, download = False)

#cifar10 is a subclass of torch.utils.data.Dataset
print("\ntype of CIFAR dataset:", type(cifar10))
print("number of items in the training set:", len(cifar10))

#with {uu}getitem{uu}, we can use the standard subscript for indexing tuples and lists to access individual items.
img_PIL, label = cifar10[99]    #cifar10[99] is a tuple, not a pure tensor and hs no attributes like 'shape'
print("image:", img_PIL, "label:", label)
#convert the PIL image to a PyTorch tensor before we can do anything with it. torchvision.transforms module defines a set of composable function-like objects that can be passed as an argument to a torchvision dataset such as datasets.CIFAR10(...).

#remember: that perform transformations on the date after it is loaded but before it is returned by {uu}getitem{uu}

#Normalizing Data
tensor_cifar10_data = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.ToTensor())
tensor_cifar10_data_val = datasets.CIFAR10(data_path, train = False, download = False, transform = transforms.ToTensor())
print("\ntensor_cifar10_data object:", tensor_cifar10_data) #<class 'torchvision.datasets.cifar.CIFAR10'>
print("\ntensor_cifar10_data_val object:", tensor_cifar10_data_val)

#While the values in the original PIL image ranged from 0 to 255 (8-bit per channel), the ToTensor transform turned the data into 32-bit floating point per channel, scaling values down from 0.0 to 1.0, and convert the shape from HxWxC to CxHxW.

#Up to now, we need to stack all tensors returned by the dataset along an extra dimension (without label):
imgs_tensor = torch.stack([img_t for img_t, _ in tensor_cifar10_data], dim = 3) #default dim = 0
imgs_tensor_val = torch.stack([img_t for img_t, _ in tensor_cifar10_data_val], dim = 3)

#Now we can easily compute the mean per channel:
print("\nshape of img train set:", imgs_tensor.shape)
mean_channel = imgs_tensor.view(3, -1).mean(dim = 1).numpy()
print("mean along channels:", mean_channel)
std_channel = imgs_tensor.view(3, -1).std(dim = 1).numpy()
print("std along channels:", std_channel)
mean_channel_val = imgs_tensor_val.view(3, -1).mean(dim = 1).numpy()
std_channel_val = imgs_tensor_val.view(3, -1).std(dim = 1).numpy()
print(mean_channel_val, std_channel_val)

#pass tuple to Normalize(), do NOT convert tensors directly to tuples!!!
transformed_cifar10_data = datasets.CIFAR10(data_path,
                                train = True,
                                download = False,
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    tuple(mean_channel),
                                    tuple(std_channel)
                                    )]))

transformed_cifar10_val_data = datasets.CIFAR10(data_path,
                                train = False,
                                download = False,
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    tuple(mean_channel_val),
                                    tuple(std_channel_val)
                                    )]))
print("\nthe transformed validation set:", transformed_cifar10_val_data)
#Note that, at this point, plotting an image drawn from the dataset won’t provide us with a faithful representation of the actual image, This is because normalization has shifted the RGB levels outside the 0.0 to 1.0 range and changed the overall magnitudes of the channels.

#distinguishing birds from airplanes
label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2_data = [(img, label_map[label]) for img, label in transformed_cifar10_data if label in [0, 2]]
cifar2_val_data = [(img, label_map[label]) for img, label in transformed_cifar10_val_data if label in [0, 2]]

#a fully connected classifier

#plot
img_bird, label_bird = cifar2_data[0]
#plt.imshow(img_bird.permute(1,2,0)) ##we had to use permute to change the order of the axes from CxHxW to HxWxC to match what Matplotlib expects.
#plt.show()

#a loss for classifying
#use nn.LogSoftmax instead of nn.softmax to make calculation numerically stable
nn_model = nn.Sequential(
             nn.Linear(3072, 512),
             nn.Tanh(),
             nn.Linear(512, 2),
             nn.LogSoftmax(dim = 1))    ##add a softmax at the end of our model, and our network will be equipped at producing probabilities.

#instantiate negative log likelihood loss:
loss_fn = nn.NLLLoss()

#the loss takes the output of nn.LogSoftmax for a batch as the first argument and a tensor of class indices (0's and 1's in our case) as the second one.

#training loop
train_loader = torch.utils.data.DataLoader(cifar2_data,
                                            batch_size = 64,
                                            shuffle = True)
val_loader = torch.utils.data.DataLoader(cifar2_val_data,
                                            batch_size = 64,
                                            shuffle = False)
learning_rate = 1e-2
optimizer = optim.SGD(nn_model.parameters(), lr = learning_rate)
n_epochs = 100

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        out = nn_model(imgs.view(batch_size, -1))
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {:2d}, Loss: {:6.6f}".format(epoch, float(loss)))

#performance evaluation
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        out = nn_model(imgs.view(batch_size, -1))
        _, index = torch.max(out, dim = 1)
        total += labels.shape[0]
        correct += int((index == labels).sum())
print('\nAccuracy:', float(correct / total))
