import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import math
from alae.alae_ffhq_inference import decode
import ot as pot
from functools import partial
from typing import Optional
from torch.distributions.multivariate_normal import MultivariateNormal


"""Utility samplers and evaluation helpers for LightSBB experiments."""


class TensorSampler:
    """Sampler wrapper for an in-memory tensor dataset."""

    def __init__(self, tensor, device='cuda'):
        """Store the tensor on the target device for fast random sampling."""
        self.device = device
        self.tensor = torch.clone(tensor).to(device)

    def sample(self, size=5):
        """Sample unique items without replacement.

        Args:
            size: Number of elements to sample.

        Returns:
            Tensor batch with shape ``(size, ...)``.
        """
        assert size <= self.tensor.shape[0]

        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False),
                           device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)


class GeneratorTwoD:
    """Synthetic 2D data generator used by benchmark scripts."""

    def __init__(self, dataset, dim=2, device="cpu"):
        """Initialize a generator for ``normal``, ``moons`` or ``8gaussians`` data."""
        self.dim = dim
        self.device = device
        self.dataset = dataset

    def sample(self, n):
        """Sample ``n`` points from the configured dataset family."""
        if self.dataset == 'moons':
            return self.sample_moons(n)
        elif self.dataset == "8gaussians":
            return self.sample_8gaussians(n)
        else:
            return self.sample_normal(n)

    def sample_normal(self, n):
        """Sample isotropic Gaussian points in ``dim`` dimensions."""
        m = MultivariateNormal(torch.zeros(self.dim).to(self.device), torch.eye(self.dim).to(self.device))
        return m.sample((n,))

    def eight_normal_sample(self, n, scale=1, var=1.):
        """Sample from a mixture of eight Gaussians arranged on a circle."""
        m = MultivariateNormal(
            torch.zeros(self.dim).to(self.device), math.sqrt(var) * torch.eye(self.dim).to(self.device)
        )
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = torch.tensor(centers, device=self.device) * scale
        noise = m.sample((n,))
        multi = torch.multinomial(torch.ones(8), n, replacement=True).to(self.device)
        data = []
        for i in range(n):
            data.append(centers[multi[i]] + noise[i])
        data = torch.stack(data)
        return data

    def sample_8gaussians(self, n):
        """Sample the benchmark eight-Gaussians dataset variant."""
        return self.eight_normal_sample(n, scale=5, var=0.1).float()

    def sample_moons(self, n):
        """Sample moons dataset."""
       x0, _ = generate_moons(n, noise=0.2)
       return x0 * 3 - 1


def wasserstein(
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: Optional[str] = None,
        reg: float = 0.05,
        power: int = 2,
        **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def get_indepedent_plan_sample_fn(sampler_x, sampler_y):
    """Create a callable that draws independent source/target batches."""
    def ret_fn(batch_size):
        """Return one minibatch from each sampler."""
        x_samples = sampler_x(batch_size)
        y_samples = sampler_y(batch_size)
        return x_samples, y_samples

    return ret_fn


def sample_alae_beta_large(model, alae_model, beta, x_inds_test, test_inp_images, test_latents,
                number_of_samples=3, device='cpu', SEED=42, n_pictures=10):
    """Generate and decode mapped ALAE latents for large-beta models.

    Returns decoded mapped-to-target images, round-tripped source images, and
    the original input images selected for visualization.
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    inds_to_map = np.random.choice(np.arange((x_inds_test < 300).sum()), size=n_pictures, replace=False)

    y_mapped_all = []
    x_mapped_all = []
    latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]])

    inp_images = test_inp_images[x_inds_test[inds_to_map]]

    for k in range(number_of_samples):
        y_latent_to_map = latent_to_map.to(device) - 1 / beta * model.get_drift(
            torch.zeros(len(latent_to_map), device=device), latent_to_map.to(device))
        y_mapped = model(y_latent_to_map.to(device))
        y_mapped_all.append(y_mapped)

        T = (torch.ones(len(y_mapped)) * (1 - 1e-2)).to(device)
        x_mapped = y_mapped + 1 / beta * model.get_drift(T, y_mapped)
        x_mapped_all.append(x_mapped)

    y_mapped = torch.stack(y_mapped_all, dim=1)
    x_mapped = torch.stack(x_mapped_all, dim=1)

    y_decoded_all = []
    x_decoded_all = []
    with torch.no_grad():
        for k in range(number_of_samples):
            decoded_img = decode(alae_model, y_mapped[:, k].cpu())
            decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(
                torch.uint8).permute(
                0, 2, 3, 1).numpy()
            y_decoded_all.append(decoded_img)

            decoded_img = decode(alae_model, x_mapped[:, k].cpu())
            decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(
                torch.uint8).permute(
                0, 2, 3, 1).numpy()
            x_decoded_all.append(decoded_img)

    y_decoded_all = np.stack(y_decoded_all, axis=1)
    x_decoded_all = np.stack(x_decoded_all, axis=1)

    return y_decoded_all, x_decoded_all, inp_images


def sample_alae(model, model_inv, alae_model, beta, x_inds_test, test_inp_images, test_latents,
                    number_of_samples=3, device='cpu', SEED=42, n_pictures=10):
    """Generate and decode mapped ALAE latents using model and inverse map.

    Returns decoded mapped-to-target images, round-tripped source images, and
    the original input images selected for visualization.
    """
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    inds_to_map = np.random.choice(np.arange((x_inds_test < 300).sum()), size=n_pictures, replace=False)

    y_mapped_all = []
    x_mapped_all = []
    latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]])

    inp_images = test_inp_images[x_inds_test[inds_to_map]]

    for k in range(number_of_samples):
        y_latent_to_map = model_inv(torch.zeros((len(latent_to_map), 1), device=device),
                                    latent_to_map)
        y_mapped = model(y_latent_to_map.to(device))
        y_mapped_all.append(y_mapped)

        T = (torch.ones(len(y_mapped)) * (1 - 1e-2)).to(device)
        x_mapped = y_mapped + 1 / beta * model.get_drift(T, y_mapped)
        x_mapped_all.append(x_mapped)

    y_mapped = torch.stack(y_mapped_all, dim=1)
    x_mapped = torch.stack(x_mapped_all, dim=1)

    y_decoded_all = []
    x_decoded_all = []
    with torch.no_grad():
        for k in range(number_of_samples):
            decoded_img = decode(alae_model, y_mapped[:, k].cpu())
            decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(
                torch.uint8).permute(
                0, 2, 3, 1).numpy()
            y_decoded_all.append(decoded_img)

            decoded_img = decode(alae_model, x_mapped[:, k].cpu())
            decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(
                torch.uint8).permute(
                0, 2, 3, 1).numpy()
            x_decoded_all.append(decoded_img)

    y_decoded_all = np.stack(y_decoded_all, axis=1)
    x_decoded_all = np.stack(x_decoded_all, axis=1)

    return y_decoded_all, x_decoded_all, inp_images
