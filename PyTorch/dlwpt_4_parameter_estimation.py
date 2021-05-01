import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   #for ApplePersistenceIgnoreState
from matplotlib import pyplot as plt

#PyTorch is designed to make it easy to create models for which the derivatives of the fitting error, with respect to the parameters, can be expressed analytically.

###4.1.1 Learning is parameter estimation
#A hot problem
t_celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_units   = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_celsius = torch.tensor(t_celsius)
t_units   = torch.tensor(t_units)
t_units_rescaled = 0.1 * t_units    #a simple normalization

#Measure of error: Loss(Cost) Function
#Convex: grow monotonically as the predicted value moves farther from the true value in either direction.
def model(t_u, w, b):
    return w * t_u + b

def loss_fun(t_p, t_c): #mean_squared_loss
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

#broadcasting rules:
#1. Each tensor has at least one dimension;
#2. When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist. (In place operations do not allow in-place tensor to change shape as a result of the broadcast.)

#4.1.6 Getting analytical
def dloss_d_tp(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs

def dtp_dw(t_u, w, b):
    return t_u

def dtp_db(t_u, w, b):
    return 1.0

def grad_fun(t_u, t_p, t_c, w, b):  #chain rule
    dloss_dw = dloss_d_tp(t_p, t_c) * dtp_dw(t_u, w, b)
    dloss_db = dloss_d_tp(t_p, t_c) * dtp_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])  #concatenate

#4.1.7 The training loop
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p  = model(t_u, *params)  #Q^(s,a,w) = neural network(w)
        loss = loss_fun(t_p, t_c)   #t_c(label): r + Gamma*Q^(s',a',w)
        grad = grad_fun(t_u, t_p, t_c, *params)
        params = params - grad * learning_rate
        print("Epoch:{:3} Loss Function: {}".format(epoch, loss.float()))
        print("Params:{} Gradient:{}".format(params, grad))
    return params

parameters = training_loop(
                n_epochs = 5000,
                learning_rate = 1e-2,
                params = torch.tensor([1.0, 0.0]),
                t_u = t_units_rescaled,
                t_c = t_celsius)  #to limit the magnitude of learning_rate * grad, you could simply choose a smaller learning_rate

#matplotlib inline
t_predicted = model(t_units_rescaled, *parameters)  #remember that you are training on the normalized unknown units but you are plotting the raw unknown values
fig = plt.figure(dpi = 600)
plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')
plt.plot(t_units, t_predicted, label = "Linear Regression")

#Q: what's detach? Returns a new tensor, detached from the current graph, the result will never require gradient.

plt.plot(t_units, t_celsius, 'o', label = "Raw Data")
plt.legend()
plt.savefig("../DL_PyTorch/Figures/temprature_regression.png")
plt.close()
