import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def training_sbb_beta_large(sampler_x, sampler_y, model, beta, K, lr=1e-3, n_epochs=10000, batch_size=512, eps=0.1, 
                            safe_t=1e-2, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")
        curr_epoch = max(5000, int(n_epochs * np.exp(-0.2 * k)))

        for epoch in range(curr_epoch):

            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)

            if k == 0:
                y_0_ = x_0_.clone()
                y_T_ = x_T_.clone()
            else:
                t_0 = torch.zeros(1, device=device)
                y_0_ = x_0_ - 1 / beta * model.get_drift(t_0, x_0_)  # B, d
                t_N = torch.ones(len(x_0_), device=device) * (1 - safe_t)
                y_T_ = x_T_ - 1 / beta * model.get_drift(t_N, x_T_)  # B, d

            t = torch.FloatTensor(len(y_0_), 1).uniform_(0, 1 - safe_t).to(device)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)  # B, d
            predicted_drift = model.get_drift(t.squeeze(), y_t)  # B, d
            target_drift = (y_T_ - y_t) / (1 - t)

            loss = F.mse_loss(target_drift, predicted_drift)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(
                    f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item():.4f}')

    model = model.eval()
    return model
