"""OT-CFM baseline: minibatch optimal-transport conditional flow matching."""

import torch
import torch.nn as nn
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from tqdm import tqdm


class VectorField(nn.Module):
    """Time-conditioned MLP predicting the flow-matching velocity field."""

    def __init__(self, input_dim, d_model=1024, t_model=128):
        """Initialize the velocity network.

        Args:
            input_dim: Dimensionality of the latent space.
            d_model: Hidden width for the state encoder.
            t_model: Hidden width for the time encoder.
        """
        super().__init__()

        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_model),
            nn.LayerNorm(t_model),
            nn.GELU(),
            nn.Linear(t_model, t_model)
        )

        self.x_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model + t_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, input_dim)
        )

    def forward(self, t, x):
        """Predict the velocity at time ``t`` and state ``x``.

        Args:
            t: (n,) or (n, 1) times in [0, 1].
            x: (n, input_dim) states.

        Returns:
            (n, input_dim) predicted velocities.
        """
        if t.dim() == 1:
            t = t[:, None]
        h = torch.cat([self.t_encoder(t), self.x_encoder(x)], dim=-1)
        return self.decoder(h)


class OTCFM:
    """Deterministic ODE transport trained by conditional flow matching."""

    name = "otcfm"

    @staticmethod
    def add_arguments(parser):
        """Register the hyperparameters this baseline exposes on the CLI."""
        parser.add_argument("--n-iters", type=int, default=20000)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--sigma", type=float, default=0.0)
        parser.add_argument("--d-model", type=int, default=1024)

    def __init__(self, input_dim, d_model=1024, device="cuda"):
        """Build the velocity network on ``device``."""
        self.input_dim = input_dim
        self.device = device
        self.net = VectorField(input_dim=input_dim, d_model=d_model).to(device)

    @classmethod
    def from_args(cls, input_dim, args, device="cuda"):
        """Build the baseline from parsed CLI arguments."""
        return cls(input_dim=input_dim, d_model=args.d_model, device=device)

    def train(self, X_sampler, Y_sampler, args):
        """Fit the velocity field on the source and target training samplers.

        Args:
            X_sampler: Source sampler exposing ``sample(n)``.
            Y_sampler: Target sampler exposing ``sample(n)``.
            args: Parsed CLI arguments carrying the hyperparameters.
        """
        matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=args.sigma)
        opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        self.net.train()
        for _ in tqdm(range(args.n_iters)):
            x_0 = X_sampler.sample(args.batch_size).to(self.device)
            x_1 = Y_sampler.sample(args.batch_size).to(self.device)

            t, x_t, u_t = matcher.sample_location_and_conditional_flow(x_0, x_1)
            loss = ((self.net(t, x_t) - u_t) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def transport(self, x_0, n_steps=100):
        """Integrate the learned field from t=0 to t=1 with midpoint steps.

        Args:
            x_0: (n, input_dim) source latents.
            n_steps: Number of integration steps.

        Returns:
            (n, input_dim) transported latents on ``self.device``.
        """
        self.net.eval()
        x = x_0.to(self.device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((len(x),), i * dt, device=self.device)
            x_mid = x + 0.5 * dt * self.net(t, x)
            x = x + dt * self.net(t + 0.5 * dt, x_mid)

        return x

    def checkpoint(self):
        """Return a picklable dict holding the weights and the architecture."""
        return {
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "d_model": self.net.x_encoder[0].out_features,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, input_dim, device="cuda"):
        """Rebuild a trained baseline from ``checkpoint``."""
        model = cls(input_dim=input_dim, d_model=checkpoint["d_model"], device=device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.to(device)
        return model
