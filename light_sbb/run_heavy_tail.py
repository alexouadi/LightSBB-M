"""
Heavy-tail benchmark: run LightSBB on 1D distributions where KL(mu_T || Wiener) = +inf.

In Schrödinger Bridge theory, the optimal coupling requires finite KL divergence
w.r.t. the reference Wiener measure.  This script stress-tests LightSBB on targets
that violate this condition, so we can compare its behaviour to Light SB.

After training, 1000 samples drawn from the learned model are saved to a .npy file.
"""

import os
import numpy as np
import torch

from lightsbm import LightSBM, MLP_network
from train_lightsbb import training_sbb
from train_lightsbb_beta_large import training_sbb_beta_large
from utils import HeavyTailSampler1D


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dim = 1
n_potentials = 50
S_init = 0.1
batch_size = 512
K = 15

SEED = np.random.randint(0, 100000)
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"SEED: {SEED}")

# --- pick an experiment ---
# mu_0  (source): 'gaussian' or 'dirac'
# mu_T  (target): 'student_2', 'student_1', 'cauchy', 'pareto', 'lognormal'
source_name = "gaussian"
target_name = "student_2"

eps = 1
beta = 10

X_sampler = HeavyTailSampler1D(source_name, device=device)
Y_sampler = HeavyTailSampler1D(target_name, device=device)

experiment_tag = f"src-{source_name}_tgt-{target_name}_b{beta}_e{eps}"
print(f"Experiment: {experiment_tag}")

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

model = LightSBM(
    dim=dim,
    n_potentials=n_potentials,
    epsilon=eps,
    S_diagonal_init=S_init,
    is_diagonal=True,
)
model.to(device)

init_samples_x = X_sampler.sample(n_potentials // 2)
init_samples_y = Y_sampler.sample(n_potentials - n_potentials // 2)
init_samples = torch.cat([init_samples_x, init_samples_y], dim=0)
model.init_r_by_samples(init_samples)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if beta >= 100:
    model = training_sbb_beta_large(
        X_sampler, Y_sampler, model, beta,
        K=K, n_epochs=20000, min_epoch=20000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, print_every=2000, device=device,
    )
else:
    model_inv = MLP_network(input_dim=dim, t_model=8, d_model=32)
    model_inv.to(device)

    model, model_inv = training_sbb(
        X_sampler, Y_sampler, model, model_inv, beta,
        K=K, n_epochs=20000, min_epoch=20000, batch_size=batch_size,
        lr=1e-3, eps=eps, safe_t=1e-2, print_every=2000, device=device,
    )

# ---------------------------------------------------------------------------
# Generate 1000 samples from the trained model and save to .npy
# ---------------------------------------------------------------------------

n_samples = 1000
X_0 = X_sampler.sample(n_samples)

model.eval()

if K > 1:
    if beta >= 100:
        t_zeros = torch.zeros(n_samples, device=device)
        Y_0 = (X_0 - 1 / beta * model.get_drift(t_zeros, X_0)).detach()
    else:
        with torch.no_grad():
            Y_0 = model_inv(torch.zeros((n_samples, 1), device=device), X_0)

    Y_T_sbb = model(Y_0)
    T = torch.ones(n_samples, device=device) * (1 - 1e-2)
    X_T_sbb = (Y_T_sbb + 1 / beta * model.get_drift(T, Y_T_sbb)).detach()
else:
    X_T_sbb = model(X_0)

samples_np = X_T_sbb.cpu().numpy()   # shape (n_samples, 1)

out_path = f"samples_{experiment_tag}_seed{SEED}.npy"
np.save(out_path, samples_np)
print(f"\nSaved {n_samples} samples → {os.path.abspath(out_path)}")
print(f"Array shape: {samples_np.shape}")


# rng = np.random.default_rng(0)
# samples_a = rng.standard_normal(1000)          # Gaussian
# samples_b = t.rvs(df=2, size=1000, random_state=rng)  # Student-t(2)
# plot_distributions(samples_a, samples_b, label1="Gaussian", label2="Student-t (df=2)")