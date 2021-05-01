import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sn
sn.set_style('darkgrid')

# build network
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, output_dim),
                        )

    def forward(self, x):
        return self.network(x)

# data input
batch_size, input_dim, hidden_dim, output_dim = 32, 1, 256, 1
x = torch.randn(batch_size, input_dim)
y = -x**2

# model initialization
network = Network(input_dim, hidden_dim, output_dim)

# optimizer and loss function
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 1000
loss_plot = np.zeros(epochs)

# training loop
for epoch in range(epochs):
    pre = network(x)
    loss = criterion(pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_plot[epoch] = loss

# plot
x_range = torch.arange(-2, 2, 0.02).unsqueeze(1)
y_pred = network(x_range)

plt.figure(dpi=500)
plt.subplot(2,1,1)
plt.plot(loss_plot, color='green')
plt.ylabel('loss')
plt.subplot(2,1,2)
plt.plot(x_range, y_pred.detach(), label="neural network regression")
plt.plot(x, y, 'o', label = 'raw data')
plt.plot(x, network(x).detach(), 'kx')
plt.ylabel('raw data')
plt.legend()
plt.savefig('neuralRegression.png')
plt.close()
