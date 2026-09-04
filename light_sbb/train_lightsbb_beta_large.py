import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


"""Training loop for LightSBB in the large-beta regime."""


def training_sbb_beta_large(sampler_x, sampler_y, model, beta, K, lr=1e-3, n_epochs=10000, min_epoch=2000,
                            batch_size=512, eps=0.1, safe_t=1e-2, print_every=2000, device='cpu'):
    """Train LightSBB when beta is large enough to avoid learning an inverse map.

    Args:
        sampler_x: Source sampler exposing ``sample(batch_size)``.
        sampler_y: Target sampler exposing ``sample(batch_size)``.
        model: ``LightSBM`` instance to optimize.
        beta: Correction coefficient used for latent updates.
        K: Number of outer bridge refinement stages.
        lr: Optimizer learning rate.
        n_epochs: Base epochs per stage before decay schedule.
        min_epochs: Minimum number of epochs per stage.
        batch_size: Minibatch size.
        eps: Diffusion variance scale.
        safe_t: Margin to avoid numerical instability at ``t=1``.
        print_every: Number of training epochs between progress prints. 
        device: Training device.

    Returns:
        Trained ``model`` in evaluation mode.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")
        curr_epoch = max(min_epoch, int(n_epochs * np.exp(-0.2 * k)))

        for epoch in range(curr_epoch):

            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)

            if k == 0:
                y_0_ = x_0_.clone()
                y_T_ = x_T_.clone()
            else:
                t_zeros = torch.zeros(len(x_0_), device=device)
                t_ones = torch.ones(len(x_T_), device=device) * (1 - safe_t)
                y_0_ = (x_0_ - (1.0 / beta) * model.get_drift(t_zeros, x_0_)).detach()  # B, d
                y_T_ = (x_T_ - (1.0 / beta) * model.get_drift(t_ones, x_T_)).detach()  # B, d

            t = torch.FloatTensor(len(y_0_), 1).uniform_(0, 1 - safe_t).to(device)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)  # B, d
            predicted_drift = model.get_drift(t.squeeze(), y_t)  # B, d
            target_drift = (y_T_ - y_t) / (1 - t)

            loss = F.mse_loss(target_drift, predicted_drift)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(
                    f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item():.4f}')

    model = model.eval()
    return model
