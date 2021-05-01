#1.local operations on neighborhoods
#2.translation-invariance
#3.models with a lot fewer parameters
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

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
std_channel = imgs_tensor.view(3, -1).std(dim = 1).numpy()
mean_channel_val = imgs_tensor_val.view(3, -1).mean(dim = 1).numpy()
std_channel_val = imgs_tensor_val.view(3, -1).std(dim = 1).numpy()

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

#load model
loaded_model = Net()
loaded_model.load_state_dict(torch.load(save_path + 'bird_vs_airplane.pt'))

for loader in [train_loader, val_loader]:
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            val_outputs = loaded_model(imgs)
            _, index = torch.max(val_outputs, dim = 1)
            total += labels.shape[0]
            correct += int((index == labels).sum())
    print('Accuracy:', correct / total)
