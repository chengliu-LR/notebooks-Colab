#1.local operations on neighborhoods
#2.translation-invariance
#3.models with a lot fewer parameters
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from torchsummary import summary
matplotlib.use('Agg')
import datetime

data_path = '../DL_PyTorch/data-cifar10/'
save_path = '../DL_PyTorch/trained_models/'
tensor_cifar10_data = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.ToTensor())
tensor_cifar10_data_val = datasets.CIFAR10(data_path, train = False, download = False, transform = transforms.ToTensor())

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
                                    tuple(std_channel),
                                    )]))

transformed_cifar10_val_data = datasets.CIFAR10(data_path,
                                train = False,
                                download = False,
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    tuple(mean_channel_val),
                                    tuple(std_channel_val),
                                    )]))

#distinguishing birds from airplanes
label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']
cifar2_data = [(img, label_map[label]) for img, label in transformed_cifar10_data if label in [0, 2]]
cifar2_val_data = [(img, label_map[label]) for img, label in transformed_cifar10_val_data if label in [0, 2]]

train_loader = torch.utils.data.DataLoader(cifar2_data,
                                            batch_size = 64,
                                            shuffle = True)
val_loader = torch.utils.data.DataLoader(cifar2_val_data,
                                            batch_size = 64,
                                            shuffle = False)

conv = nn.Conv2d(3, 1, kernel_size = 3, padding = 1)

with torch.no_grad():
    conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0],
                                   [-1.0, 0.0, 1.0]])
                                   #weight has three 3*3 tensors
    conv.bias.zero_()

print(conv)
print(conv.weight, conv.bias)

img_bird, _ = cifar2_data[0]
output = conv(img_bird.unsqueeze(0))
print(img_bird.unsqueeze(0).shape, output.shape)

#plt.imshow(output[0, 0], cmap = 'gray')    #Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
#plt.imshow(output.detach(), cmap = 'gray') #Invalid shape (1, 1, 32, 32) for image data
print(output[0,0].shape)
plt.imshow(output[0,0].detach(), cmap = 'gray')
plt.savefig("../DL_PyTorch/Figures/bird_conv.png", dpi=300)

#Max Pooling
#leave nn.Sequential for something that gives us more flexibility.
#before feeding data to linear fully connected model, we need a reshape step nn.MaxPool2d.view()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)   #8 channels, 8x8 pixels
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
        
summary(Net(), (3,32,32))

#training loop
def training_loop(n_epochs, optimizer, model, criterion, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {:2}, Training Loss {:2.6f}'.format(datetime.datetime.now(), epoch, float(loss_train)))

model = Net()
learning_rate = 1e-2
optimizer = optim.SGD(params = model.parameters(), lr = learning_rate)

training_loop(n_epochs = 100,
              optimizer = optimizer,
              model = model,
              criterion = nn.CrossEntropyLoss(),  #equivalent to LogSoftmax & NLLLoss
              train_loader = train_loader)

#save model
torch.save(model.state_dict(), save_path + 'bird_vs_airplane.pt')
#Training accuracy: 0.9323
#Testing Accuracy: 0.8865
