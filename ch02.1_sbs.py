import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

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