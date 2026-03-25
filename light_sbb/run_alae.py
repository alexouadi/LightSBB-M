import torch
import numpy as np
import os
import pickle
from lightsbm import LightSBM, MLP_network
from utils import TensorSampler
from train_lightsbb_beta_large import training_sbb_beta_large
from train_lightsbb import training_sbb

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.isdir('model'):
    os.makedirs('model')

if not os.path.isdir('model_inv'):
    os.makedirs('model_inv')
        
dim = 512
eps = 0.1
n_potentials = 10
S_init = 0.1
beta = 0.8
batch_size = 512
K = 5

path = f"b{beta}_e{eps}.pkl"
save_path = os.path.join("model",path)
print(save_path)

input_data = 'ADULT'
target_data = 'CHILDREN'

train_size = 60000
test_size = 10000

latents = np.load("data/latents.npy")
gender = np.load("data/gender.npy")
age = np.load("data/age.npy")

train_latents, test_latents = latents[:train_size], latents[train_size:]
train_gender, test_gender = gender[:train_size], gender[train_size:]
train_age, test_age = age[:train_size], age[train_size:]

x_inds_train = np.arange(train_size)[
    (train_age >= 18).reshape(-1) * (train_age != -1).reshape(-1)
    ]
x_inds_test = np.arange(test_size)[
    (test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
x_data_train = train_latents[x_inds_train]
x_data_test = test_latents[x_inds_test]

y_inds_train = np.arange(train_size)[
    (train_age < 18).reshape(-1) * (train_age != -1).reshape(-1)
    ]
y_inds_test = np.arange(test_size)[
    (test_age < 18).reshape(-1) * (test_age != -1).reshape(-1)
    ]
y_data_train = train_latents[y_inds_train]
y_data_test = test_latents[y_inds_test]

X_train = torch.tensor(x_data_train)
Y_train = torch.tensor(y_data_train)

X_test = torch.tensor(x_data_test)
Y_test = torch.tensor(y_data_test)


X_sampler = TensorSampler(X_train, device=device)
Y_sampler = TensorSampler(Y_train, device=device)

model = LightSBM(dim=dim,
                 n_potentials=n_potentials,
                 epsilon=eps,
                 S_diagonal_init=S_init,
                 is_diagonal=True)

model.to(device)

if beta >= 100:
    model = training_sbb_beta_large(
        X_sampler, Y_sampler, model, beta, K=K, n_epochs=10000, min_epoch=5000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, device=device
    )

else:
    model_inv = MLP_network(
        input_dim=dim,
        t_model=32,
        d_model=128
    )
    model_inv.to(device)

    model, model_inv = training_sbb(
        X_sampler, Y_sampler, model, model_inv, beta, K=K, n_epochs=10000, min_epoch=5000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, device=device
    )
    
    model_inv.to('cpu')
    save_path_inv = os.path.join("model_inv",path)
    with open(save_path_inv, "wb") as f:
        pickle.dump(model_inv.state_dict(), f)

model.to('cpu')
with open(save_path, "wb") as f:
    pickle.dump(model.state_dict(), f)

print(save_path, device)
