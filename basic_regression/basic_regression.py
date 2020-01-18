
import numpy as np
import torch

NUM_TEST_VECS = 300

def create_train_data(m, c):
    X = 100*np.random.rand(NUM_TEST_VECS)
    Y = np.array([m*x+c for x in X])
    e = np.random.normal(scale=1.5, size=len(X))
    Y = Y +e
    return X,Y

def estimate(X, Y):
    Y = torch.tensor(Y, dtype=torch.float32)
    X_1 = torch.stack([torch.ones(len(X)),
             torch.tensor(X, dtype=torch.float32)])
    W = torch.ones(2, requires_grad=True)

    for i in range(200000):
        L = torch.norm(Y - torch.matmul(W, X_1))
        print(f"loss={L}")
        L.backward()
        with torch.no_grad():
            W -= 1e-5 * W.grad
            W.grad.zero_()

        print(W)

if __name__ == "__main__":
    X,Y = create_train_data(2,3)
    estimate(X,Y)

    
