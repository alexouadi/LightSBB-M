"""LightSB-M baseline: the Schrodinger bridge, on the shared ALAE evaluation.

LightSB-M has no volatility penalty and no Bass map, so its volatility is
sqrt(eps) I by construction. Training is the large-beta LightSBB loop at K = 1,
whose ``k == 0`` branch never reads beta, which is what makes this the Schrodinger
bridge rather than an SBB run with a particular beta.
"""

import torch

from lightsbm import LightSBM
from train_lightsbb_beta_large import training_sbb_beta_large

from .lightsbb import PRINT_EVERY


class LightSB:
    """Schrodinger bridge trained by the K = 1 branch that never applies beta."""

    name = "lightsb"

    @staticmethod
    def add_arguments(parser):
        """Register the hyperparameters this baseline exposes on the CLI."""
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument("--n-potentials", type=int, default=10)
        parser.add_argument("--s-init", type=float, default=0.1)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--n-epochs", type=int, default=10000)
        parser.add_argument("--min-epoch", type=int, default=5000)
        parser.add_argument("--safe-t", type=float, default=1e-2)
        parser.add_argument("--print-every", type=int, default=PRINT_EVERY)

    def __init__(self, input_dim, eps=0.1, n_potentials=10, s_init=0.1, device="cuda"):
        """Build the bridge model on ``device``."""
        self.input_dim = input_dim
        self.eps = eps
        self.n_potentials = n_potentials
        self.s_init = s_init
        self.device = device

        self.model = LightSBM(dim=input_dim,
                              n_potentials=n_potentials,
                              epsilon=eps,
                              S_diagonal_init=s_init,
                              is_diagonal=True)
        self.model.to(device)

    @classmethod
    def from_args(cls, input_dim, args, device="cuda"):
        """Build the baseline from parsed CLI arguments."""
        return cls(input_dim=input_dim, eps=args.eps, n_potentials=args.n_potentials,
                   s_init=args.s_init, device=device)

    def train(self, X_sampler, Y_sampler, args):
        """Fit the bridge with the single stage that carries no Bass correction.

        Beta is passed only because the loop takes it; at K = 1 the ``k == 0``
        branch never reads it, so its value cannot affect the result.

        Args:
            X_sampler: Source sampler exposing ``sample(n)``.
            Y_sampler: Target sampler exposing ``sample(n)``.
            args: Parsed CLI arguments carrying the hyperparameters.
        """
        self.model = training_sbb_beta_large(
            X_sampler, Y_sampler, self.model, beta=1.0, K=1,
            n_epochs=args.n_epochs, min_epoch=args.min_epoch,
            batch_size=args.batch_size, lr=args.lr, eps=args.eps,
            safe_t=args.safe_t, print_every=args.print_every, device=self.device
        )

    @torch.no_grad()
    def transport(self, x_0, n_steps=None):
        """Map source latents through the bridge.

        With no Bass map the source already lives in bridge space, so it is pushed
        straight through without the drift correction LightSBB-M applies first.

        Args:
            x_0: (n, input_dim) source latents.
            n_steps: Unused; kept so every baseline shares one call signature.

        Returns:
            (n, input_dim) transported latents on ``self.device``.
        """
        self.model.eval()
        return self.model(x_0.to(self.device))

    def checkpoint(self):
        """Return a picklable dict holding the weights and the architecture."""
        return {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "eps": self.eps,
            "n_potentials": self.n_potentials,
            "s_init": self.s_init,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, input_dim, device="cuda"):
        """Rebuild a trained baseline from ``checkpoint``."""
        model = cls(input_dim=input_dim, eps=checkpoint["eps"],
                    n_potentials=checkpoint["n_potentials"],
                    s_init=checkpoint["s_init"], device=device)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.to(device)
        return model
