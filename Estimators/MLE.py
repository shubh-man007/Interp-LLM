import torch
from .base import BaseIntrinsicEstimator
from tqdm import tqdm
import numpy as np
import gc

class MLE(BaseIntrinsicEstimator):
    def __init__(
        self,
        k1,
        k2,
        mackay=False,
        batch_size=None,
        device="cpu",
    ):
        super().__init__(batch_size, device)
        self.k1 = k1
        self.k2 = k2
        self.mackay = mackay

    def fit(self, data):
        nearest_distances = self.nearestNeighbours(self.k2 + 1, data)
        torch.cuda.empty_cache()
        gc.collect()

        return self._fitallMLE(nearest_distances)

    @torch.no_grad()
    def _fitallMLE(self, nearest_distances, verbose=True, get_local=False):
        if verbose:
            if self.mackay:
                print("Getting MLE Corrected Predictions")
            else:
                print("Getting MLE Predictions")

        dimensions_ = []

        mle_correct_dimensions_ = []

        if get_local:
            local_dimensions = {}

        for k in tqdm(range(self.k1, self.k2 + 1), desc="MLE"):
            nearest_distances_ = nearest_distances[:, : k + 1]
            dim, dim_corr, local_dim = self._fitMLE(nearest_distances_)
            dimensions_.append(dim)
            mle_correct_dimensions_.append(dim_corr)

            if get_local:
                local_dimensions[f"id_k={k}"] = local_dim

        dimension_ = torch.mean(torch.tensor(dimensions_)).item()

        if self.mackay is None:
            mle_correct_dimensions_ = torch.mean(
                torch.tensor(mle_correct_dimensions_)
            ).item()

        if verbose:
            if self.mackay is None:
                print("#" * 100, "MLE Prediction: ", dimension_, "#" * 100)
                print(
                    "#" * 100,
                    "MLE Corrected Prediction: ",
                    mle_correct_dimensions_,
                    "#" * 100,
                )
            if self.mackay:
                print("#" * 100, "MLE Corrected Prediction: ", dimension_, "#" * 100)
            else:
                print("#" * 100, "MLE Prediction: ", dimension_, "#" * 100)

        if self.mackay is None:
            dimension_ = (dimension_, mle_correct_dimensions_)

        if get_local:
            return dimension_, local_dimensions
        return dimension_

    @torch.no_grad()
    def _fitMLE(self, nearest_distances):
        ratio = nearest_distances[:, :-1] / (nearest_distances[:, -1].reshape(-1, 1) + 1e-8)
        log_ratio = torch.log(ratio + 1e-8)
        dk_x = -1 / (torch.mean(log_ratio, dim=1) + 1e-8)

        if self.mackay is None:
            dk_x_inverse_mean = torch.mean(1 / (dk_x + 1e-8))
            mle_corrected = 1 / (dk_x_inverse_mean + 1e-8)
            dimension_ = torch.mean(dk_x)

            return dimension_, mle_corrected, dk_x
        if self.mackay:
            dk_x_inverse_mean = torch.mean(1 / (dk_x + 1e-8))
            dimension_ = 1 / (dk_x_inverse_mean + 1e-8)
        else:
            dimension_ = torch.mean(dk_x)

        return dimension_, None, dk_x
