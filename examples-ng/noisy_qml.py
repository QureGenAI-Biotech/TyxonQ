"""
QML task on noisy PQC with vmapped Monte Carlo noise simulation
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "true"
# one need this for pytorch+gpu combination in some cases
import time
import torch
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")

# numpy data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data

# Load MNIST data
mnist_train = MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Convert to numpy for processing
x_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()
x_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

x_train = x_train[..., np.newaxis] / 255.0


def filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y


datapoints = 64
batch = 16
logfile = "qml_param_v2.npy"
n = 9
m = 4
maxiter = 200
data_preparation = "v1"

x_train, y_train = filter_pair(x_train, y_train, 0, 1)

if data_preparation == "v1":
    # Resize using torch
    x_train_tensor = torch.from_numpy(x_train).float()
    x_train_resized = torch.nn.functional.interpolate(
        x_train_tensor.permute(0, 3, 1, 2), 
        size=(int(np.sqrt(n)), int(np.sqrt(n))), 
        mode='bilinear', 
        align_corners=False
    ).permute(0, 2, 3, 1).numpy()
    x_train = np.array(x_train_resized > 0.5, dtype=np.float32)
    x_train = np.squeeze(x_train).reshape([-1, n])

else:  # "v2"
    from sklearn.decomposition import PCA

    x_train = PCA(n).fit_transform(x_train.reshape([-1, 28 * 28]))


# Create PyTorch dataset
class MNISTDataset(data.Dataset):
    def __init__(self, x, y, maxiter):
        self.x = torch.from_numpy(x[:datapoints]).float()
        self.y = torch.from_numpy(y[:datapoints]).float()
        self.maxiter = maxiter
        
    def __len__(self):
        return self.maxiter
        
    def __getitem__(self, idx):
        # Return random batch
        indices = torch.randperm(len(self.x))[:batch]
        return self.x[indices], self.y[indices]

mnist_data = MNISTDataset(x_train, y_train, maxiter)


def f(param, seed, x, pn):
    c = tq.Circuit(n)
    px, py, pz = pn, pn, pn
    for i in range(n):
        if data_preparation == "v1":
            # 避免在vmap中使用标量theta
            theta_val = x[i] * np.pi / 2
            c.rx(i, theta=theta_val)
        else:
            theta_val = torch.atan(x[i])
            c.rx(i, theta=theta_val)
    for j in range(m):
        for i in range(n - 1):
            c.cx(i, i + 1)
            c.depolarizing(i, px=px, py=py, pz=pz, status=seed[j, i, 0])
            c.depolarizing(i + 1, px=px, py=py, pz=pz, status=seed[j, i, 1])
        for i in range(n):
            c.rz(i, theta=param[j, i, 0])
        for i in range(n):
            c.rx(i, theta=param[j, i, 1])

    ypreds = torch.stack([tq.backend.real(c.expectation_ps(z=[i])) for i in range(n)])

    return torch.mean(ypreds)


vf = tq.utils.append(K.vmap(f, vectorized_argnums=1), torch.mean)


def loss(param, scale, seeds, x, y, pn):
    # Ensure single-sample feature vector for circuit building
    if hasattr(x, 'dim') and x.dim() > 1:
        x = x[0]
        y = y[0]
    ypred = vf(param, seeds, x, pn)
    ypred = torch.sigmoid(scale * ypred)
    y = y.to(torch.float32)
    return (
        torch.real(-y * torch.log(ypred) - (1 - y) * torch.log(1 - ypred)),
        ypred,
    )


def acc(yps, ys):
    if hasattr(yps, 'detach'):
        yps = yps.detach().cpu().numpy()
    if hasattr(ys, 'detach'):
        ys = ys.detach().cpu().numpy()
    yps = (np.sign(yps - 0.5) + 1) / 2
    return 1 - np.mean(np.logical_xor(ys, yps))


vgloss = K.jit(K.vvag(loss, argnums=(0, 1), vectorized_argnums=(3, 4), has_aux=True))
vloss = K.jit(K.vmap(loss, vectorized_argnums=(3, 4)))


def train(param=None, scale=None, noise=0, noc=1, fixed=True, val_step=40):
    times = []
    val_times = []
    if param is None:
        param = torch.nn.Parameter(torch.randn(m, n, 2))
    if scale is None:
        scale = torch.tensor(15.0, dtype=torch.float32)
    else:
        scale = torch.tensor(scale, dtype=torch.float32)
    
    optimizer = torch.optim.Adam([param], lr=1e-2)
    optimizer2 = torch.optim.Adam([scale], lr=5e-2)
    
    pn = noise * torch.ones([], dtype=torch.float32)

    try:
        for i in range(maxiter):
            xs, ys = mnist_data[i]
            seeds = torch.rand(noc, m, n, 2)
            time0 = time.time()
            
            # Forward pass
            loss_val, ypred = loss(param, scale, seeds, xs, ys, pn)
            
            # Backward pass
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss_val.backward()
            
            time1 = time.time()
            times.append(time1 - time0)
            
            optimizer.step()
            if fixed is False:
                optimizer2.step()
                
            if i % val_step == 0:
                print("%s round" % str(i))
                print("scale: ", scale.item())
                time0 = time.time()
                inference(param)
                time1 = time.time()
                val_times.append(time1 - time0)
                if len(times) > 1:
                    print("batch running time est.: ", np.mean(times[1:]))
                    print("batch staging time est.: ", times[0])
                if len(val_times) > 1:
                    print("full set running time est.: ", np.mean(val_times[1:]))
                    print("full set staging time est.: ", val_times[0])
    except KeyboardInterrupt:
        pass

    np.save(logfile, param.detach().cpu().numpy())
    return param


def inference(param=None, scale=None, noise=0, noc=1, debug=False):
    pn = noise * torch.ones([], dtype=torch.float32)
    if param is None:
        param = torch.from_numpy(np.load(logfile))
    if scale is None:
        scale = torch.tensor(15.0, dtype=torch.float32)
    else:
        scale = torch.tensor(scale, dtype=torch.float32)
    seeds = torch.rand(noc, m, n, 2)
    
    with torch.no_grad():
        vs, yps = vloss(
            param,
            scale,
            seeds,
            torch.from_numpy(x_train[:datapoints]).float(),
            torch.from_numpy(y_train[:datapoints]).float(),
            pn,
        )
    if debug:
        print(yps)
    print("loss: ", torch.mean(vs))
    print("acc on training: %.4f" % acc(yps, y_train[:datapoints]))


if __name__ == "__main__":
    # run a short training to validate functionality
    train(noise=0.005, scale=10, noc=8, fixed=True, val_step=50)
    # inference(noise=0.01, noc=50, scale=15, debug=True)
