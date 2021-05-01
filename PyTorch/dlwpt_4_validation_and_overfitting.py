import torch
import torch.optim as optim
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

NORM_FACTOR = 0.1
t_celsius = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_units = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_units_rescaled = NORM_FACTOR * t_units
def linear_regression(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()
#divide sample set to training set and validation set
n_samples = t_units.shape[0]
n_valid = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)    #random permutation
train_indices = shuffled_indices[:-n_valid]
valid_indices = shuffled_indices[-n_valid:]

train_t_units_rescaled = NORM_FACTOR * t_units[train_indices]
valid_t_units_rescaled = NORM_FACTOR * t_units[valid_indices]

train_t_celsius = t_celsius[train_indices]
valid_t_celsius = t_celsius[valid_indices]

#From the training loop, you only ever call *backward* on the train_loss, so errors only ever backpropagate based on the training set. The validation set is used to provide an independent evaluation of the accuracy of the model's output on data that wasn't used for training.
def training_loop(n_epochs, optimizer, params, train_t_u, train_t_c, valid_t_u, valid_t_c):
    for epoch in range(1, n_epochs):
        train_t_p = linear_regression(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)  #forward
        #switch off autograd (validation set doesn't need autograd) to speed up and save memory
        with torch.no_grad():
            valid_t_p  = linear_regression(valid_t_u, *params)
            valid_loss = loss_fn(valid_t_p, valid_t_c)
            #assert valid_loss.requires_grad == False
        optimizer.zero_grad()
        train_loss.backward()   #backward
        optimizer.step()        #optimize
        if epoch < 4 or epoch % 500 == 0:
            print("Epoch: {:5d}, Training Loss: {:2.6f}, Validation Loss: {:2.6f}".format(epoch, train_loss, valid_loss))
    return params
#training loop
params_target = torch.tensor([1.0, 0.0], requires_grad = True)
parameters_optim = training_loop(n_epochs = 6000,
                        optimizer = optim.SGD([params_target], lr = 1e-2),
                        params = params_target,
                        train_t_u = train_t_units_rescaled,
                        train_t_c = train_t_celsius,
                        valid_t_u = valid_t_units_rescaled,
                        valid_t_c = valid_t_celsius)
#plot
print(parameters_optim)
t_predicted = linear_regression(t_units_rescaled, *parameters_optim)
fig = plt.figure(dpi = 600)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_units, t_predicted.detach(), label = "Linear Regression")
plt.plot(t_units, t_celsius, 'o', label = "Raw Data")
plt.legend()
plt.title("Using Training & Validation")
plt.savefig("../DL_PyTorch/figures/temprature_training_validation.png")
plt.close()