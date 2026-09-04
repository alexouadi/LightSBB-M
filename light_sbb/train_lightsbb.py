import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


"""Training loop for LightSBB in the moderate-beta regime."""


def training_sbb(sampler_x, sampler_y, model, model_inv, beta, K, lr=1e-3, n_epochs=10000, min_epoch=5000,
                 batch_size=512, eps=0.1, safe_t=1e-2, print_every=2000, device='cpu'):
    """Train LightSBB with an auxiliary inverse model.

    This routine alternates between:
    1) updating the bridge model drift, and
    2) updating the inverse network used to pull samples into bridge space.

    Args:
        sampler_x: Source sampler exposing ``sample(batch_size)``.
        sampler_y: Target sampler exposing ``sample(batch_size)``.
        model: ``LightSBM`` instance.
        model_inv: Inverse network (typically ``MLP_network``).
        beta: Correction coefficient used in inverse consistency updates.
        K: Number of outer bridge refinement stages.
        lr: Learning rate for both optimizers.
        n_epochs: Base epochs per stage before decay schedule.
        min_epochs: Minimum number of epochs per stage.
        batch_size: Minibatch size.
        eps: Diffusion variance scale.
        safe_t: Margin to avoid numerical instability at ``t=1``.
        print_every: Number of training epochs between progress prints. 
        device: Training device.

    Returns:
        Tuple ``(model, model_inv)`` in evaluation mode.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_inv = optim.Adam(model_inv.parameters(), lr=lr)

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")
        curr_epoch = max(min_epoch, int(n_epochs * np.exp(-0.2 * k)))

        for epoch in range(curr_epoch):
            model.train()
            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)

            if k == 0:
                y_0_ = x_0_.clone()
                y_T_ = x_T_.clone()
            else:
                with torch.no_grad():
                    t_0 = torch.zeros((len(x_0_), 1), device=device)
                    y_0_ = model_inv(t_0, x_0_)
                    t_N = torch.ones((len(x_T_), 1), device=device) * (1 - safe_t)
                    y_T_ = model_inv(t_N, x_T_)

            t = torch.FloatTensor(len(y_0_), 1).uniform_(0, 1 - safe_t).to(device)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)
            
            predicted_drift = model.get_drift(t.squeeze(-1), y_t)
            target_drift = (y_T_ - y_t) / (1 - t)
 
            loss = F.mse_loss(target_drift, predicted_drift)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item()}')

        for epoch in range(curr_epoch):
            model_inv.train()
            model.eval()  
 
            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)
 
            t_0 = torch.zeros((len(x_0_), 1), device=device)
            t_N = torch.ones((len(x_T_), 1), device=device) * (1 - safe_t)
 
            Y0 = (x_0_ + 1 / beta * model.get_drift(t_0.squeeze(-1), x_0_)).detach()
            YT = (x_T_ + 1 / beta * model.get_drift(t_N.squeeze(-1), x_T_)).detach()
 
            y_0_ = model_inv(t_0, Y0)   
            y_T_ = model_inv(t_N, YT)   
            
            loss_inv = F.mse_loss(x_0_, y_0_) + F.mse_loss(x_T_, y_T_)
            optimizer_inv.zero_grad()
            loss_inv.backward()
            torch.nn.utils.clip_grad_norm_(model_inv.parameters(), 1.0)
            optimizer_inv.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss Inv: {loss_inv.item()}')

    model = model.eval()
    model_inv = model_inv.eval()
    return model, model_inv
