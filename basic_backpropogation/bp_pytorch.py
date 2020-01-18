
import numpy as np
from numpy.random import randn
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = torch.tensor(randn(N, D_in)), torch.tensor(randn(N, D_out))


W1, W2 = torch.tensor(randn(D_in, H)), torch.tensor(randn(H, D_out))

W1.requires_grad_()
W2.requires_grad_()

for t in range(2000):
    h = 1/(1+torch.exp(-torch.mm(x, W1)))
    y_pred = torch.mm(h, W2)
    loss = ((y_pred - y)**2).sum()
    loss.backward()

    print(t,loss)
    with torch.no_grad():
        W1 -= 1e-4 * W1.grad
        W2 -= 1e-4 * W2.grad
        W1.grad.zero_()
        W2.grad.zero_()
    
    
