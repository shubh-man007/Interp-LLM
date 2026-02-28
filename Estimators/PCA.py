import torch
import numpy as np

class PCA(object):
    def __init__(self, alpha=10, beta=0.8, condition="max_variance", device="cpu"):
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.condition = condition

    def fit(self, X):
        X = X.to(self.device)

        covariance_matrix = torch.cov(X.T)
        eigenvalues = torch.linalg.eigvals(covariance_matrix)
        eigenvalues = torch.real(eigenvalues)
        eigenvalues = torch.sort(eigenvalues, descending=True).values

        if self.condition == "max_variance":
            self.dimension_ = self._fitMaxVariance(eigenvalues)
        elif self.condition == "sum_variance":
            self.dimension_ = self._fitSumVariance(eigenvalues)

        return self.dimension_

    def _fitMaxVariance(self, eigenvalues):
        d = 1
        while (d < len(eigenvalues)) and (
            eigenvalues[d - 1] < self.alpha * eigenvalues[d]
        ):
            d += 1

        return d - 1

    def _fitSumVariance(self, eigenvalues):
        d = 1
        sum_eigenvalues = torch.sum(eigenvalues)
        sum_eigenvalues_d = torch.sum(eigenvalues[:d])

        while (d < len(eigenvalues)) and (
            sum_eigenvalues_d < self.beta * sum_eigenvalues
        ):
            sum_eigenvalues_d += eigenvalues[d]
            d += 1

        return d - 1
