"""[SF]^2M baseline: simulation-free Schrodinger bridges via score and flow matching."""

import torch
import torch.nn as nn
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from tqdm import tqdm

from .otcfm import VectorField


class SF2M:
    """Stochastic transport whose drift and diffusion are matched without simulation."""

    name = "sf2m"

    @staticmethod
    def add_arguments(parser):
        """Register the hyperparameters this baseline exposes on the CLI."""
        parser.add_argument("--n-iters", type=int, default=20000)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--sigma", type=float, default=0.1)
        parser.add_argument("--d-model", type=int, default=1024)

    def __init__(self, input_dim, d_model=1024, sigma=0.1, device="cuda"):
        """Build the drift and score networks on ``device``.

        Args:
            input_dim: Dimensionality of the latent space.
            d_model: Hidden width of both networks.
            sigma: Diffusion coefficient of the bridge; also fixes the entropic
                regularisation of the coupling.
            device: Torch device the networks live on.
        """
        self.input_dim = input_dim
        self.sigma = sigma
        self.device = device
        self.net = VectorField(input_dim=input_dim, d_model=d_model).to(device)
        self.score_net = VectorField(input_dim=input_dim, d_model=d_model).to(device)

    @classmethod
    def from_args(cls, input_dim, args, device="cuda"):
        """Build the baseline from parsed CLI arguments."""
        return cls(input_dim=input_dim, d_model=args.d_model, sigma=args.sigma,
                   device=device)

    def train(self, X_sampler, Y_sampler, args):
        """Fit the drift and score networks on the source and target samplers.

        Args:
            X_sampler: Source sampler exposing ``sample(n)``.
            Y_sampler: Target sampler exposing ``sample(n)``.
            args: Parsed CLI arguments carrying the hyperparameters.
        """
        matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=args.sigma,
                                                          ot_method="exact")
        opt = torch.optim.Adam(
            list(self.net.parameters()) + list(self.score_net.parameters()), lr=args.lr
        )

        self.net.train()
        self.score_net.train()
        for _ in tqdm(range(args.n_iters)):
            x_0 = X_sampler.sample(args.batch_size).to(self.device)
            x_1 = Y_sampler.sample(args.batch_size).to(self.device)

            t, x_t, u_t, eps = matcher.sample_location_and_conditional_flow(
                x_0, x_1, return_noise=True
            )
            lambda_t = matcher.compute_lambda(t)

            flow_loss = ((self.net(t, x_t) - u_t) ** 2).mean()
            score_loss = (
                (lambda_t[:, None] * self.score_net(t, x_t) + eps) ** 2
            ).mean()

            opt.zero_grad()
            (flow_loss + score_loss).backward()
            opt.step()

    @torch.no_grad()
    def transport(self, x_0, n_steps=100):
        """Integrate the learned SDE from t=0 to t=1 with Euler-Maruyama steps.

        Args:
            x_0: (n, input_dim) source latents.
            n_steps: Number of integration steps.

        Returns:
            (n, input_dim) transported latents on ``self.device``.
        """
        self.net.eval()
        self.score_net.eval()
        x = x_0.to(self.device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((len(x),), i * dt, device=self.device)
            # The score network carries the whole correction to the ODE drift, so no
            # sigma^2/2 factor multiplies it here.
            drift = self.net(t, x) + self.score_net(t, x)
            noise = torch.randn_like(x)
            x = x + dt * drift + self.sigma * (dt ** 0.5) * noise

        return x

    def checkpoint(self):
        """Return a picklable dict holding the weights and the architecture."""
        return {
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "score_state_dict": {
                k: v.cpu() for k, v in self.score_net.state_dict().items()
            },
            "d_model": self.net.x_encoder[0].out_features,
            "sigma": self.sigma,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, input_dim, device="cuda"):
        """Rebuild a trained baseline from ``checkpoint``."""
        model = cls(input_dim=input_dim, d_model=checkpoint["d_model"],
                    sigma=checkpoint["sigma"], device=device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.score_net.load_state_dict(checkpoint["score_state_dict"])
        model.net.to(device)
        model.score_net.to(device)
        return model
