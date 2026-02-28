import torch
from .base import BaseIntrinsicEstimator
from tqdm import tqdm
import numpy as np
import gc

class Gride(BaseIntrinsicEstimator):
    def __init__(
        self,
        k1,
        k2,
        discard_fraction=0.1,
        batch_size=None,
        device="cpu",
        patience=100,
        learning_rate=0.1,
    ):
        super().__init__(batch_size, device)
        self.k1 = k1
        self.k2 = k2
        self.discard_fraction = discard_fraction
        self.patience = patience
        self.learning_rate = learning_rate

    def fit(self, data):
        nearest_distances = self.nearestNeighbours(self.k2, data)
        torch.cuda.empty_cache()
        gc.collect()

        return self._fitGride(nearest_distances)

    def _fitGride(self, nearest_distances, verbose=True):
        if verbose:
            print("Getting GRIDE Predictions")

        N = len(nearest_distances)
        mu_ = nearest_distances[:, self.k2 - 1] / nearest_distances[:, self.k1 - 1]
        mu_.requires_grad_(False)
        mu_ = mu_.to(self.device)

        d = torch.tensor(1.0, requires_grad=True, device=self.device)
        optimizer = torch.optim.SGD([d], lr=self.learning_rate)

        patience = 0
        iterations = 0
        best_loss = float("inf")

        while patience < self.patience:
            optimizer.zero_grad()
            loss = self._grideLoss(d, N, mu_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(d, 0.1)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1

            iterations += 1
            print(
                f"Loss: {loss.item():.4f} | Patience: {patience}/{self.patience}| Iterations: {iterations}| Best Loss: {best_loss:.4f}| d: {d.item():.4f}",
                end="\r",
            )

        print()

        if verbose:
            print("#" * 100, "Gride Prediction: ", d.item(), "#" * 100)
        return d.item()

    def _grideLoss(self, d, N, mu_):
        loss = -N * torch.log(d + 1e-8)
        loss -= (self.k2 - self.k1 + 1) * torch.sum(
            torch.log(torch.pow(mu_, d) - 1 + 1e-8)
        )
        loss += (((self.k2 - 1) * d) + 1) * torch.sum(torch.log(mu_ + 1e-8))

        return loss
