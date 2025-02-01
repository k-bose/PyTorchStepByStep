import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim


def build_train_step(model, loss_fn, optimizer):
    def perform_train_step(x, y):
        model.train()
        # step 1 - compute model's prediction
        yhat = model(x)
        # step 2 - compute the loss
        loss = loss_fn(yhat, y)
        # step 3 - compute the gradients
        loss.backward()
        # step 4 - update the parameters
        optimizer.step()
        optimizer.zero_grad()
        # return the loss value
        return loss.item()
    return perform_train_step

def build_val_step(model, loss_fn):
    def perform_val_step(x, y):
        model.eval()
        # step 1 - compute model's prediction
        yhat = model(x)
        # step 2 - compute the loss
        loss = loss_fn(yhat, y)
        # return the loss value
        return loss.item()
    return perform_val_step

def mini_batch(device, data_loader, step):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    return np.mean(mini_batch_losses)


# Data Generation
n_total = 100
true_b = 1
true_w = 2

np.random.seed(42)
x = np.random.rand(n_total, 1)
epsilon = 0.1 * np.random.randn(n_total, 1)

y = true_b + true_w * x + epsilon

# Data Preparation
# build tensor from numpy array
# convert 64-bit to 32-bit tensor
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

# build dataset with all data points
dataset = TensorDataset(x_tensor, y_tensor)

# perform the train-val data split
ratio = 0.8
n_train = int(n_total * ratio)
n_val = n_total - n_train

torch.manual_seed(13)
train_data, val_data = random_split(dataset, [n_train, n_val])

# build training and validation dataloader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)


# Model Configuration
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set learning rate
lr = 0.1

# define model
torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(1, 1))

# define loss function
loss_fn = nn.MSELoss(reduction='mean')

# define optimizer
optimizer = optim.SGD(model.parameters(), lr)

# create step function for training and validating our model
train_step = build_train_step(model, loss_fn, optimizer)
val_step = build_val_step(model, loss_fn)


# Model Training
# set number of epoch
n_epochs = 200

# placeholders for training and validation loss values
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    train_loss = mini_batch(device, train_loader, train_step)
    train_losses.append(train_loss)
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)

print(model.state_dict())