import math
import torch
import torch.nn as nn
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal


class LightSBM(nn.Module):
    def __init__(self, dim=2, n_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=0.1):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        self.log_alpha = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.r_scale = nn.Parameter(torch.ones(n_potentials, 1))

        self.S_log_diagonal_matrix = nn.Parameter(torch.log(S_diagonal_init * torch.ones(n_potentials, self.dim)))
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(n_potentials, self.dim, self.dim)
        )

    def init_r_by_samples(self, samples):

        assert samples.shape[0] == self.r.shape[0]

        self.r.data = torch.clone(samples.to(self.r.device))

    def get_S(self):
        if self.is_diagonal:
            S = torch.exp(self.S_log_diagonal_matrix)
        else:
            S = (self.S_rotation_matrix * (torch.exp(self.S_log_diagonal_matrix))[:, None, :]) @ torch.permute(
                self.S_rotation_matrix, (0, 2, 1))
        return S

    def get_r(self):
        return self.r

    @torch.no_grad()
    def forward(self, x):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon

        log_alpha = self.log_alpha

        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size // sampling_batch_size if batch_size % sampling_batch_size == 0 else (
                                                                                                    batch_size // sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size * i:sampling_batch_size * (i + 1)]

            if self.is_diagonal:
                x_S_x = (sub_batch_x[:, None, :] * S[None, :, :] * sub_batch_x[:, None, :]).sum(dim=-1)
                x_r = (sub_batch_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :] * sub_batch_x[:, None, :]
            else:
                x_S_x = (sub_batch_x[:, None, None, :] @ (S[None, :, :, :] @ sub_batch_x[:, None, :, None]))[:, :, 0, 0]
                x_r = (sub_batch_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, :, :] @ sub_batch_x[:, None, :, None])[:, :, :, 0]

            exp_argument = (x_S_x + 2 * x_r) / (2 * epsilon) + log_alpha[None, :]

            if self.is_diagonal:
                mix = Categorical(logits=exp_argument)
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon * S)[None, :, :]), 1)
                gmm = MixtureSameFamily(mix, comp)

            else:
                mix = Categorical(logits=exp_argument)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon * S)
                gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples

    def get_drift(self, t, x):
        x = x.clone().detach().requires_grad_(True)
        epsilon = self.epsilon
        r = self.get_r()

        S_diagonal = torch.exp(self.S_log_diagonal_matrix)  # shape: potential*dim
        A_diagonal = (t / (epsilon * (1 - t)))[:, None, None] + 1 / (epsilon * S_diagonal)[None, :, :]  # B*K*dim

        S_log_det = torch.sum(self.S_log_diagonal_matrix, dim=-1)  # shape: potential
        A_log_det = torch.sum(torch.log(A_diagonal), dim=-1)  # shape: batch*potential

        log_alpha = self.log_alpha  # shape: potential

        S = S_diagonal  # shape: potential*dim
        A = A_diagonal  # shape: batch*potential*dim
        S_inv = 1 / S  # shape: potential*dim
        A_inv = 1 / A  # shape: batch*potential*dim
        c = ((1 / (epsilon * (1 - t)))[:, None] * x)[:, None, :] + (
                                                                           r / (epsilon * S_diagonal)
                                                                   )[None, :, :]  # B*K*dim
        exp_arg = (
                log_alpha[None, :] - 0.5 * S_log_det[None, :] - 0.5 * A_log_det
                - 0.5 * ((r * S_inv * r) / epsilon).sum(dim=-1)[None, :] + 0.5 * (c * A_inv * c).sum(dim=-1)
        )
        lse = torch.logsumexp(exp_arg, dim=-1)
        drift = (-x / (1 - t[:, None]) + epsilon * torch.autograd.grad(
            lse, x, grad_outputs=torch.ones_like(lse, device=lse.device), create_graph=True)[
            0])

        return drift

    def sample_euler_maruyama(self, x, n_steps):
        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1 / n_steps
        trajectory = [x]

        for i in range(n_steps):
            x = x + self.get_drift(t, x) * dt + math.sqrt(dt) * torch.sqrt(epsilon) * torch.randn_like(x,
                                                                                                       device=x.device)
            t += dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)

    def sample_at_time_moment(self, x, t):
        t = t.to(x.device)
        y = self(x)

        return t * y + (1 - t) * x + torch.sqrt(t * (1 - t) * self.epsilon) * torch.randn_like(x)


class MLP_network(torch.nn.Module):
    def __init__(self, input_dim, d_model, t_model):
        super().__init__()
        self.d_model = d_model

        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_model),
            nn.LayerNorm(t_model),
            nn.GELU(),
            nn.Linear(t_model, t_model)
        )

        self.y_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model + t_model, min(d_model, t_model)),
            # nn.LayerNorm(min(d_model, t_model)),
            nn.SiLU(),
            nn.Linear(min(d_model, t_model), input_dim)
        )

    def forward(self, t, y):
        t_embed = self.t_encoder(t)
        y_embed = self.y_encoder(y)
        y_emb = torch.cat([t_embed, y_embed], dim=-1)
        return self.decoder(y_emb)
