import numpy as np

# Data Generation
n_total = 100
true_b = 1
true_w = 2

np.random.seed(42)
x = np.random.rand(n_total, 1)
epsilon = 0.1 * np.random.randn(n_total, 1)

y = true_b + true_w * x + epsilon