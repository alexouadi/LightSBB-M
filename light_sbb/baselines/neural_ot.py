"""NOT baseline: neural optimal transport with a weak quadratic cost.

Follows the reference implementation of Korotin et al. (ICLR 2023), keeping the
max-min loop and hyperparameters of their weak notebook but swapping the image
UNet/ResNet pair for the MLPs their low-dimensional notebook uses.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def mlp(input_dim, output_dim, hidden, n_hidden=3):
    """Build the ReLU MLP stack used for both NOT networks.

    Args:
        input_dim: Input width.
        output_dim: Output width.
        hidden: Hidden width.
        n_hidden: Number of hidden layers.

    Returns:
        `nn.Sequential` with ``n_hidden`` ReLU layers.
    """
    layers = [nn.Linear(input_dim, hidden), nn.ReLU(True)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(True)]
    layers.append(nn.Linear(hidden, output_dim))
    return nn.Sequential(*layers)


def init_weights(module):
    """Apply the Kaiming initialization used by the reference implementation."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(module.bias)


class NOT:
    """Stochastic transport map trained adversarially under a weak quadratic cost."""

    name = "not"

    @staticmethod
    def add_arguments(parser):
        """Register the hyperparameters this baseline exposes on the CLI."""
        parser.add_argument("--max-steps", type=int, default=100000)
        parser.add_argument("--t-iters", type=int, default=10)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--hidden", type=int, default=1024)
        parser.add_argument("--z-dim", type=int, default=8)
        parser.add_argument("--z-size", type=int, default=8)
        parser.add_argument("--z-std", type=float, default=0.1)
        parser.add_argument("--gamma", type=float, default=0.66)
        parser.add_argument("--gamma-iters", type=int, default=25000)

    def __init__(self, input_dim, hidden=1024, z_dim=8, z_std=0.1, device="cuda"):
        """Build the transport and critic networks on ``device``."""
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.z_std = z_std
        self.device = device

        self.T = mlp(input_dim + z_dim, input_dim, hidden).to(device)
        self.f = mlp(input_dim, 1, hidden).to(device)
        self.f.apply(init_weights)

    @classmethod
    def from_args(cls, input_dim, args, device="cuda"):
        """Build the baseline from parsed CLI arguments."""
        return cls(input_dim=input_dim, hidden=args.hidden, z_dim=args.z_dim,
                   z_std=args.z_std, device=device)

    def _push(self, x, z_size):
        """Map each row of ``x`` through ``z_size`` independent noise draws.

        Args:
            x: (n, input_dim) source batch.
            z_size: Noise samples per source point.

        Returns:
            (n, z_size, input_dim) transported points.
        """
        n = len(x)
        x_rep = x[:, None].repeat(1, z_size, 1)
        z = torch.randn(n, z_size, self.z_dim, device=self.device) * self.z_std
        return self.T(torch.cat([x_rep, z], dim=-1))

    def train(self, X_sampler, Y_sampler, args):
        """Run the max-min loop of the weak quadratic cost.

        Args:
            X_sampler: Source sampler exposing ``sample(n)``.
            Y_sampler: Target sampler exposing ``sample(n)``.
            args: Parsed CLI arguments carrying the hyperparameters.
        """
        if args.z_size < 2:
            raise ValueError("the weak cost needs at least two noise draws per point")

        T_opt = torch.optim.Adam(self.T.parameters(), lr=args.lr, weight_decay=1e-10)
        f_opt = torch.optim.Adam(self.f.parameters(), lr=args.lr, weight_decay=1e-10)

        for step in tqdm(range(args.max_steps)):
            # Diversity is annealed in: gamma ramps linearly, then holds.
            gamma = args.gamma * min(1.0, step / args.gamma_iters)

            self.f.requires_grad_(False)
            for _ in range(args.t_iters):
                x = X_sampler.sample(args.batch_size).to(self.device)
                T_xz = self._push(x, args.z_size)

                # The variance term is what makes the cost weak: it pays for diversity
                # across noise draws, so the map stays one-to-many.
                T_loss = (
                    ((x - T_xz.mean(dim=1)) ** 2).mean()
                    - self.f(T_xz.flatten(0, 1)).mean()
                    + T_xz.var(dim=1).mean() * (1 - gamma - 1.0 / args.z_size)
                )

                T_opt.zero_grad()
                T_loss.backward()
                T_opt.step()

            self.f.requires_grad_(True)
            x = X_sampler.sample(args.batch_size).to(self.device)
            y = Y_sampler.sample(args.batch_size).to(self.device)
            with torch.no_grad():
                T_xz = self._push(x, 1).squeeze(1)

            f_loss = self.f(T_xz).mean() - self.f(y).mean()

            f_opt.zero_grad()
            f_loss.backward()
            f_opt.step()

    @torch.no_grad()
    def transport(self, x_0, n_steps=None):
        """Draw one transported sample per source point.

        Args:
            x_0: (n, input_dim) source latents.
            n_steps: Unused; kept so every baseline shares one call signature.

        Returns:
            (n, input_dim) transported latents on ``self.device``.
        """
        self.T.eval()
        return self._push(x_0.to(self.device), 1).squeeze(1)

    def checkpoint(self):
        """Return a picklable dict holding the weights and the architecture."""
        return {
            "T": {k: v.cpu() for k, v in self.T.state_dict().items()},
            "f": {k: v.cpu() for k, v in self.f.state_dict().items()},
            "hidden": self.T[0].out_features,
            "z_dim": self.z_dim,
            "z_std": self.z_std,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, input_dim, device="cuda"):
        """Rebuild a trained baseline from ``checkpoint``."""
        model = cls(input_dim=input_dim, hidden=checkpoint["hidden"],
                    z_dim=checkpoint["z_dim"], z_std=checkpoint["z_std"], device=device)
        model.T.load_state_dict(checkpoint["T"])
        model.f.load_state_dict(checkpoint["f"])
        model.T.to(device)
        model.f.to(device)
        return model
