import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def training_sbb(sampler_x, sampler_y, model, model_inv, beta, K, lr=1e-3, n_epochs=10000, batch_size=512,
                 eps=0.1, safe_t=1e-2, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_inv = optim.Adam(model_inv.parameters(), lr=lr)

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")
        curr_epoch = max(5000, int(n_epochs * np.exp(-0.2 * k)))

        for epoch in range(curr_epoch):
            model.train()
            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)

            if k == 0:
                y_0_ = x_0_.clone()
                y_T_ = x_T_.clone()
            else:
                t_0 = torch.zeros((len(x_0_), 1), device=device)
                y_0_ = model_inv(t_0, x_0_)  # B, d
                t_N = torch.ones((len(x_T_), 1), device=device) * (1 - safe_t)
                y_T_ = model_inv(t_N, x_T_)  # B, d

            t = torch.FloatTensor(len(y_0_), 1).uniform_(0, 1 - safe_t).to(device)
            y_t = y_T_ * t + y_0_ * (1 - t) + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(y_0_)  # B, d
            predicted_drift = model.get_drift(t.squeeze(), y_t)  # B, d
            target_drift = (y_T_ - y_t) / (1 - t)

            loss = F.mse_loss(target_drift, predicted_drift)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {loss.item()}')

        for epoch in range(curr_epoch):
            model_inv.train()
            x_0_ = sampler_x.sample(batch_size).to(device)
            x_T_ = sampler_y.sample(batch_size).to(device)

            t_0 = torch.zeros((len(x_0_), 1), device=device)
            Y0 = x_0_ + 1 / beta * model.get_drift(t_0[0], x_0_)  # B, d
            y_0_ = model_inv(t_0, Y0)  # B, d  should be x_0_

            t_N = torch.ones((len(x_T_), 1), device=device) * (1 - safe_t)
            YT = x_T_ + 1 / beta * model.get_drift(t_N[0], x_T_)  # B, d
            y_T_ = model_inv(t_N, YT)  # B, d  should be x_T_

            loss_inv = F.mse_loss(x_0_, y_0_)
            loss_inv += F.mse_loss(x_T_, y_T_)
            optimizer_inv.zero_grad()
            loss_inv.backward()
            optimizer_inv.step()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss Inv: {loss_inv.item()}')

    model = model.eval()
    model_inv = model_inv.eval()
    return model, model_inv
