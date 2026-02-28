import torch
import numpy as np
from .KNN import KNearestNeighbours


class BaseIntrinsicEstimator(object):
    def __init__(self, batch_size=None, device="cpu"):
        self.batch_size = batch_size
        self.device = device

    def nearestNeighbours(self, k, data):
        knn_module = KNearestNeighbours(k, self.batch_size, self.device)

        if isinstance(data, torch.Tensor):
            nearest_distances = knn_module.getKnearestNeighboursEuclideanDirect(data)
        elif isinstance(data, str):
            nearest_distances = knn_module.getKnearestNeighboursEuclideanBatchedFiles(data)
        else:
            raise ValueError("Invalid data type.")

        return nearest_distances
