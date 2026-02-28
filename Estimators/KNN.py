import gc
import os
import gzip
import torch
import numpy as np
from tqdm import tqdm

class KNearestNeighbours(object):
    def __init__(self, k, batch_size=None, device="cpu"):
        self.k = k
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def getKnearestNeighboursEuclideanDirect(self, data):
        data = data.to(self.device)
        if self.batch_size is None:
            batched_data = [data]
        else:
            batched_data = torch.split(data, self.batch_size)

        nearest_distances = []

        for batch in tqdm(batched_data, desc="Finding Nearest Neighbour Batch"):
            distances_batch = torch.cdist(
                batch, data, p=2, compute_mode="donot_use_mm_for_euclid_dist"
            )
            batch_distances, _ = torch.topk(distances_batch, self.k + 1, largest=False)

            batch_distances = batch_distances[:, 1:]
            nearest_distances.append(batch_distances)

        del batched_data, batch, distances_batch, batch_distances, _
        torch.cuda.empty_cache()
        gc.collect()

        nearest_distances = torch.cat(nearest_distances, dim=0)
        return nearest_distances

    @torch.no_grad()
    def getKnearestNeighboursEuclideanBatchedFiles(self, file_path):
        batch_files = os.listdir(file_path)
        batch_files = [batch for batch in batch_files if batch.endswith(".pt")]
        batch_files.sort()

        nearest_distances = []

        for batch_num, batch_file in tqdm(
            enumerate(batch_files), desc="Finding Nearest Neighbour Batch"
        ):
            first_batch = torch.load(os.path.join(file_path, batch_file)).to(
                self.device
            )
            batch_distance_matrix = []

            for batch in tqdm(
                batch_files, desc=f"NN Batch: {batch_num}/{len(batch_files)}"
            ):
                batch = torch.load(os.path.join(file_path, batch)).to(self.device)
                distance_matrix = torch.cdist(
                    first_batch, batch, p=2, compute_mode="donot_use_mm_for_euclid_dist"
                )
                batch_distance_matrix.append(distance_matrix)

            batch_distance_matrix = torch.cat(batch_distance_matrix, dim=1)
            batch_distances, _ = torch.topk(
                batch_distance_matrix, self.k + 1, largest=False
            )

            batch_distances = batch_distances[:, 1:]
            nearest_distances.append(batch_distances)

        del (
            batch_files,
            batch_file,
            first_batch,
            batch_distance_matrix,
            batch_distances,
            _,
        )
        torch.cuda.empty_cache()
        gc.collect()

        nearest_distances = torch.cat(nearest_distances, dim=0)
        return nearest_distances
