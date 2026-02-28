import torch
from .base import BaseIntrinsicEstimator
import numpy as np
import gc

class TwoNN(BaseIntrinsicEstimator):
    def __init__(
        self,
        discard_fraction=0.1,
        batch_size=None,
        device="cpu",
    ):
        super().__init__(batch_size, device)
        self.discard_fraction = discard_fraction

    def fit(self, data):
        nearest_two_distances = self.nearestNeighbours(2, data)
        torch.cuda.empty_cache()
        gc.collect()

        return self._fitTwoNN(nearest_two_distances)

    @torch.no_grad()
    def _fitTwoNN(self, nearest_two_distances, verbose=True):
        if verbose:
            print("Getting TwoNN Predictions")

        N = len(nearest_two_distances)

        _mu = nearest_two_distances[:, 1] / (nearest_two_distances[:, 0] + 1e-10)
        mu = _mu[torch.argsort(_mu)][: int(N * (1 - self.discard_fraction))]

        Femp = torch.arange(len(mu)) / N

        x_axis = torch.log(mu + 1e-8).reshape(-1, 1).float().to(self.device)
        y_axis = -torch.log(1 - Femp + 1e-8).reshape(-1, 1).to(self.device)

        slope = torch.linalg.lstsq(x_axis, y_axis)[0].to(self.device)

        if verbose:
            print("#" * 100, "TwoNN Prediction: ", slope.item(), "#" * 100)

        return slope.item()
