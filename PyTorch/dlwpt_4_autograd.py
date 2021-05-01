import torch
import torch.optim as optim
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Agg')

NORM_FACTOR = 0.1   #normalization

###4.2 Pytorch's autograd: Backpropagate all things
##after this part, you will know what's going on under the hood:
#1. backpropagation to estimate gradients;
#2. autograd;
#3. optimizing weights of models by using gradient descent or other optimizers.
#given a forward expression, no matter how nested, PyTorch provides the gradient of that expression with respect to its *input parameters* automatically
#the basic requirement is that all functions you're dealing with are differentiable analytically

t_celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_units   = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_celsius = torch.tensor(t_celsius)
t_units   = torch.tensor(t_units)
t_units_rescaled = NORM_FACTOR * t_units    #a simple normalization

def linear_regression(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

params_target = torch.tensor([1.0, 0.0], requires_grad = True) #the target to update
#requires_grad: telling PyTorch to track the entire family tree of tensors resulting from operations on params. In case these functions are differentiable (and most PyTorch tensor operations are), the value of the derivative is automatically populated as a *grad* attribute of the *params* tensor.

##handy parameter update
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_() #this could be done at any point in the loop prior to calling loss.backward()
        t_p = linear_regression(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        #calling backward leads derivatives to *accumulate* at leaf nodes. You need to *zero the gradient explicitly after using it for parameter updates*. The reason is to provide more flexibility and control for working with gradients in complicated models.
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        #Q: what's detach? Returns a new tensor, detached from the current graph, the result will never require gradient. (Cut the back-propagation)
        if epoch % 500 == 0:
            print("Handy:{:4} {:2.6f}".format(epoch, loss.float()))
    return params

parameters = training_loop(
                n_epochs = 6000,
                learning_rate = 1e-2,
                params = params_target,
                t_u = t_units_rescaled,
                t_c = t_celsius)

##optimizer/each optimizer exposes two methods: zero_grad and step
def training_loop_optim(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = linear_regression(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print("  SGD:{:4} {:2.6f}".format(epoch, loss.float()))
    return params

parameters_optim = training_loop_optim(n_epochs = 6000,
                                    optimizer = optim.SGD([params_target], lr = 1e-2), #careful with the object (params_target, NOT params)
                                    params = params_target,
                                    t_u = t_units_rescaled,
                                    t_c = t_celsius)
                                    
##plot/remember that you are training on the normalized unknown units but you are plotting the raw unknown values
t_predicted = linear_regression(t_units_rescaled, *parameters_optim)
fig = plt.figure(dpi = 600)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_units, t_predicted.detach(), label = "Linear Regression")
#Error: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
#plt.plot(t_units.numpy(), t_predicted.detach().numpy(), label = "Linear Regression")
plt.plot(t_units, t_celsius, 'o', label = "Raw Data")
plt.legend()
plt.title("Linear Regression with Autograd")
plt.savefig("../DL_PyTorch/Figures/temprature_regression_autograd.png")
plt.close()
