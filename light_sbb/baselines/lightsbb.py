"""LightSBB-M wrapped in the baseline interface so it runs the shared evaluation.

Training is the code of ``run_alae.py``, hyperparameters included; only the plumbing
differs, so this row is produced by exactly the pipeline that measures the competing
methods (same split, same held-out latents, same decoding and scoring).
"""

import torch

from lightsbm import LightSBM, MLP_network
from train_lightsbb import training_sbb
from train_lightsbb_beta_large import training_sbb_beta_large

# run_alae.py switches to the loop that needs no inverse network at this beta.
LARGE_BETA = 100.0

# A stage runs 10000 epochs decaying to a floor of 5000, so this leaves two prints
# per stage of each loop.
PRINT_EVERY = 4000


class LightSBB:
    """Joint drift and volatility bridge, trained by the SBB loop of the paper."""

    name = "lightsbb"

    @staticmethod
    def add_arguments(parser):
        """Register the hyperparameters this baseline exposes on the CLI."""
        parser.add_argument("--beta", type=float, default=0.8)
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument("--n-potentials", type=int, default=10)
        parser.add_argument("--s-init", type=float, default=0.1)
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--n-epochs", type=int, default=10000)
        parser.add_argument("--min-epoch", type=int, default=5000)
        parser.add_argument("--safe-t", type=float, default=1e-2)
        parser.add_argument("--print-every", type=int, default=PRINT_EVERY)

    def __init__(self, input_dim, beta=0.8, eps=0.1, n_potentials=10, s_init=0.1,
                 device="cuda"):
        """Build the bridge, and the inverse network the moderate-beta loop needs."""
        self.input_dim = input_dim
        self.beta = beta
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

        self.model_inv = None
        if beta < LARGE_BETA:
            self.model_inv = MLP_network(input_dim=input_dim, t_model=32, d_model=128)
            self.model_inv.to(device)

    @classmethod
    def from_args(cls, input_dim, args, device="cuda"):
        """Build the baseline from parsed CLI arguments."""
        return cls(input_dim=input_dim, beta=args.beta, eps=args.eps,
                   n_potentials=args.n_potentials, s_init=args.s_init, device=device)

    def train(self, X_sampler, Y_sampler, args):
        """Run the SBB training loop matching the beta regime.

        Args:
            X_sampler: Source sampler exposing ``sample(n)``.
            Y_sampler: Target sampler exposing ``sample(n)``.
            args: Parsed CLI arguments carrying the hyperparameters.
        """
        if self.model_inv is None:
            self.model = training_sbb_beta_large(
                X_sampler, Y_sampler, self.model, self.beta, K=args.k,
                n_epochs=args.n_epochs, min_epoch=args.min_epoch,
                batch_size=args.batch_size, lr=args.lr, eps=args.eps,
                safe_t=args.safe_t, print_every=args.print_every, device=self.device
            )
        else:
            self.model, self.model_inv = training_sbb(
                X_sampler, Y_sampler, self.model, self.model_inv, self.beta, K=args.k,
                n_epochs=args.n_epochs, min_epoch=args.min_epoch,
                batch_size=args.batch_size, lr=args.lr, eps=args.eps,
                safe_t=args.safe_t, print_every=args.print_every, device=self.device
            )

    @torch.no_grad()
    def transport(self, x_0, n_steps=None):
        """Map source latents to the target, following ``utils.sample_alae``.

        The bridge acts on Y-space, so the source is pulled in first: through the
        inverse network at moderate beta, through the drift correction above it.
        The child is the bridge output itself, not the round trip back to X-space
        that the plotting helper also returns.

        Args:
            x_0: (n, input_dim) source latents.
            n_steps: Unused; kept so every baseline shares one call signature.

        Returns:
            (n, input_dim) transported latents on ``self.device``.
        """
        self.model.eval()
        x = x_0.to(self.device)

        if self.model_inv is None:
            t_0 = torch.zeros(len(x), device=self.device)
            # get_drift differentiates through its own input, so it needs grad
            # enabled even at inference.
            with torch.enable_grad():
                y_0 = (x - 1.0 / self.beta * self.model.get_drift(t_0, x)).detach()
        else:
            self.model_inv.eval()
            t_0 = torch.zeros((len(x), 1), device=self.device)
            y_0 = self.model_inv(t_0, x)

        return self.model(y_0)

    def checkpoint(self):
        """Return a picklable dict holding the weights and the architecture."""
        state = {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "beta": self.beta,
            "eps": self.eps,
            "n_potentials": self.n_potentials,
            "s_init": self.s_init,
        }
        if self.model_inv is not None:
            state["state_dict_inv"] = {k: v.cpu()
                                       for k, v in self.model_inv.state_dict().items()}
        return state

    @classmethod
    def from_checkpoint(cls, checkpoint, input_dim, device="cuda"):
        """Rebuild a trained baseline from ``checkpoint``."""
        model = cls(input_dim=input_dim, beta=checkpoint["beta"], eps=checkpoint["eps"],
                    n_potentials=checkpoint["n_potentials"],
                    s_init=checkpoint["s_init"], device=device)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.to(device)

        if model.model_inv is not None:
            model.model_inv.load_state_dict(checkpoint["state_dict_inv"])
            model.model_inv.to(device)
        return model
