import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from collections import OrderedDict        
from torchsummary import summary 
NORM_FACTOR = 0.1
t_celsius = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]).unsqueeze(1)
t_units = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)

#divide sample set to training set and validation set
n_samples = t_units.shape[0]
n_valid = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)    #random permutation
train_indices = shuffled_indices[:-n_valid]
valid_indices = shuffled_indices[-n_valid:]
#normalization
train_t_units_rescaled = NORM_FACTOR * t_units[train_indices]
valid_t_units_rescaled = NORM_FACTOR * t_units[valid_indices]

train_t_celsius = t_celsius[train_indices]
valid_t_celsius = t_celsius[valid_indices]

#From the training loop, you only ever call *backward* on the train_loss, so errors only ever backpropagate based on the training set. The validation set is used to provide an independent evaluation of the accuracy of the model's output on data that wasn't used for training.
def training_loop(n_epochs, optimizer, model, loss_fn, train_t_u, train_t_c, valid_t_u, valid_t_c):
    for epoch in range(1, n_epochs):
        train_t_p = model(train_t_u)
        train_loss = loss_fn(train_t_p, train_t_c)  #forward
        #switch off autograd (validation set doesn't need autograd) to speed up and save memory
        with torch.no_grad():
            valid_t_p  = model(valid_t_u)
            valid_loss = loss_fn(valid_t_p, valid_t_c)
            assert valid_loss.requires_grad == False
        optimizer.zero_grad()
        train_loss.backward()   #backward
        optimizer.step()        #optimize
        if epoch < 4 or epoch % 200 == 0:
            print("Epoch: {:5d}, Training Loss: {:2.6f}, Validation Loss: {:2.6f}".format(epoch, train_loss, valid_loss))

#use nn Modules
#use neural networks
#using OrderedDict here to ensure the ordering of the layers and emphasize that the order of the layers matters.
#all Pytorch-provided subclasses of nn.Module have their *call* method defined, which allows you to instantiate an nn.Module and call it as though it were a function, as in the following listing.
#calling an instance of nn.Module with a set of arguments ends up calling a method named *forward* with the same arguments.
#Variable: requires_grad default = False;
#Module: requires_grad default = True.
seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1,26)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(26,1))
    ]))
summary(seq_model, (1,1))
print("neural network:", seq_model)

#you can also get to the particular parameter by accessing submodules as though they were attributes:
print("output_linear.bias:", seq_model.output_linear.bias)

#print(params.shape for params in seq_model.parameters())
#feel the difference below!!!!!
print("seq_model parameters:", [params.shape for params in seq_model.parameters()])

#when you are inspecting parameters of a model made up for several submodules, it's handy to be able to identify parameters by their names.
for name, params in seq_model.named_parameters():
    print(name, params.shape)

#optimizer
#self.parameters() returns an iterator over module parameters and is typically passed to an optimizer
optimizer = optim.SGD(seq_model.parameters(), lr = 1e-3)    #the learning rate has dropped a bit to help with stability

#training loop
training_loop(n_epochs = 4000,
                optimizer = optimizer,
                model = seq_model,
                loss_fn = nn.MSELoss(),
                train_t_u = train_t_units_rescaled,
                train_t_c = train_t_celsius,
                valid_t_u = valid_t_units_rescaled,
                valid_t_c = valid_t_celsius)

#plot
t_range = torch.arange(20.0, 90.0).unsqueeze(1)
t_predicted = seq_model(NORM_FACTOR * t_range)
fig = plt.figure(dpi = 600)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_range, t_predicted.detach(), label = "neural network regression")
plt.plot(t_units, t_celsius, 'o', label = "raw data")
plt.plot(t_units, seq_model(NORM_FACTOR * t_units).detach(), 'kx')
plt.legend()
plt.title("nn.Module")
plt.savefig("./figures/temprature_nn_Model.png")
plt.close()
