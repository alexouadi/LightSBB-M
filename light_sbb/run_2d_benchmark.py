from lightsbm import LightSBM, MLP_network
from train_lightsbb import training_sbb
from train_lightsbb_beta_large import training_sbb_beta_large
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dim = 2
n_potentials = 50
S_init = 0.1
batch_size = 512
K = 10

SEED = np.random.randint(0, 100000)
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f'SEED: {SEED}')


dataset = "M8G"

if dataset == "M8G":
    X_sampler, Y_sampler = (GeneratorTwoD("moons", dim, device),
                            GeneratorTwoD("8gaussians", dim, device),
                            )
    eps = 5
    beta = 100
    
elif dataset == "N8G":
    X_sampler, Y_sampler = (GeneratorTwoD("normal", dim, device),
                            GeneratorTwoD("8gaussians", dim, device),
                            )
    eps = 1
    beta = 10
    
else:  # NM
    X_sampler, Y_sampler = (GeneratorTwoD("normal", dim, device),
                            GeneratorTwoD("moons", dim, device),
                            )
    eps = 1
    beta = 100
    K = 25

path = f'b{beta}_{dataset}_e{eps}.pkl'
print(path)

model = LightSBM(dim=dim,
                 n_potentials=n_potentials,
                 epsilon=eps,
                 S_diagonal_init=S_init,
                 is_diagonal=True)

model.to(device)

init_samples_x = X_sampler.sample(n_potentials // 2).to(device)
init_samples_y = Y_sampler.sample(n_potentials - n_potentials // 2).to(device)
init_samples = torch.cat([init_samples_x, init_samples_y], dim=0)
model.init_r_by_samples(init_samples)

if beta >= 100:
    model = training_sbb_beta_large(
        X_sampler, Y_sampler, model, beta, K=K, n_epochs=20000, min_epoch=20000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, print_every=2000, device=device
    )

else:

    model_inv = MLP_network(
        input_dim=dim,
        t_model=8,
        d_model=32
    )
    model_inv.to(device)

    model, model_inv = training_sbb(
        X_sampler, Y_sampler, model, model_inv, beta, K=K, n_epochs=20000, min_epoch=20000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, print_every=2000, device=device
    )

X_0 = X_sampler.sample(10000).to(device)
X_T = Y_sampler.sample(10000).to(device)
 
if K > 1:
    if beta >= 100:
        t_zeros = torch.zeros(len(X_0), device=device)
        Y_0 = (X_0 - 1 / beta * model.get_drift(t_zeros, X_0)).detach()
    else:
        with torch.no_grad():
            Y_0 = model_inv(torch.zeros((len(X_0), 1), device=device), X_0)
 
    Y_T_sbb = model(Y_0)  
    T = torch.ones(len(Y_T_sbb), device=device) * (1 - 1e-2)
    X_T_sbb = (Y_T_sbb + 1 / beta * model.get_drift(T, Y_T_sbb)).detach()
 
else:
    X_T_sbb = model(X_0)

print()
print(wasserstein(X_T_sbb, X_T))
print(path, device, SEED)
