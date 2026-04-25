import json
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from estimators.mle import MLE
from estimators.twonn import TwoNN
from estimators.gride import Gride
from tqdm import tqdm
import argparse
from utils import getSaveDirectories
from transformer_lens import HookedTransformer

def getEstimates(path, args, **kwargs):
    if args.mle:
        mle_module = MLE(
            k1=args.mle_k1,
            k2=args.mle_k2,
            mackay=None,
            batch_size=args.batch_size,
            device=args.device,
        )
    twonn_module = TwoNN(
        discard_fraction=0.1,
        batch_size=args.batch_size,
        device=args.device,
    )
    gride_module = Gride(
        k1=args.gride_k1,
        k2=args.gride_k2,
        discard_fraction=0.1,
        batch_size=args.batch_size,
        device=args.device,
        patience=10,
        learning_rate=0.1,
    )

    results = {}

    if args.mle:
        nearest_distances = twonn_module.nearestNeighbours(
            max(args.mle_k2, args.gride_k2), path
        )
        mle_pred, mle_local_dims = mle_module._fitallMLE(
            nearest_distances, get_local=True
        )

        mle_pred, mle_correct_pred = mle_pred
        results["mle"] = mle_pred
        results["mle_corrected"] = mle_correct_pred

        save_dir = kwargs.get("cache_dir", None)
        os.makedirs(save_dir, exist_ok=True)
        layer_name = kwargs.get("layer_name", None)
        layer_name = layer_name.replace("/", "_")

        mle_local_dims = {k: v.cpu().numpy() for k, v in mle_local_dims.items()}
        mle_local_dims = pd.DataFrame(mle_local_dims)
        mle_local_dims.to_csv(f"{save_dir}/{layer_name}.csv", index=False)
    else:
        nearest_distances = twonn_module.nearestNeighbours(args.gride_k2, path)

    twonn_pred = twonn_module._fitTwoNN(nearest_distances)
    gride_pred = gride_module._fitGride(nearest_distances)

    results["twonn"] = twonn_pred
    results["gride"] = gride_pred

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gride_k1", type=int, default=20, help="The k1 parameter for Gride estimator")
    parser.add_argument("--gride_k2", type=int, default=40, help="The k2 parameter for Gride estimator")
    parser.add_argument("--mle_k1", type=int, default=20, help="The k1 parameter for MLE estimator")
    parser.add_argument("--mle_k2", type=int, default=40, help="The k2 parameter for MLE estimator")
    parser.add_argument("--mle", type=bool, default=False, help="Whether to compute MLE estimates or not")
    parser.add_argument("--model_id", type=str, default="gpt2-small")
    parser.add_argument("--dataset_name", type=str, default="mmlu_STEM", help="Name of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing the dataset")
    parser.add_argument("--last_token", type=bool, default=False, help="Whether to use last token embeddings or all token embeddings")
    parser.add_argument("--relative_depth_of_sentence", type=float, default=1.0, help="The relative depth of the sentence ranges from 0 to 1")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples to use")
    parser.add_argument("--pythia", type=bool, default=False, help="Whether to use Pythia model checkpoints")
    parser.add_argument("--revision_step", type=int, default=0, help="Revision step for Pythia model")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the results")
    args = parser.parse_args()
    embeddings_dir, _, _, results_dir = getSaveDirectories(args)

    with open(os.path.join(results_dir, "llm_intrinsic_args.json"), "w") as f:
        json.dump(vars(args), f)

    layer_names = os.listdir(embeddings_dir)

    model = HookedTransformer.from_pretrained(args.model_id)
    all_layers = [name for name, _ in model.named_modules()]
    del model

    permutation = np.array([all_layers.index(name) for name in layer_names])
    index = np.argsort(permutation)
    layer_names = np.array(layer_names)[index].tolist()

    results_dir = os.path.join(results_dir, "intrinsic")
    os.makedirs(results_dir, exist_ok=True)

    twonn_estimates = []
    gride_estimates = []
    mle_estimates = []
    mle_estimates_corr = []

    for i, layer_name in enumerate(tqdm(layer_names)):
        path = os.path.join(embeddings_dir, layer_name)
        results = getEstimates(
            path,
            args,
            cache_dir=os.path.join(results_dir, "mle_local_dims"),
            layer_name=layer_name,
        )

        if args.mle:
            mle_estimates.append(results["mle"])
            mle_estimates_corr.append(results["mle_corrected"])
        twonn_estimates.append(results["twonn"])
        gride_estimates.append(results["gride"])

        if args.mle:
            print(
                f"Layer: {layer_name} || MLE: {mle_estimates[-1]} || TwoNN: {twonn_estimates[-1]} || Gride: {gride_estimates[-1]}"
            )
            results = pd.DataFrame(
                columns=["Layer", "MLE", "MLE_Corrected", "TwoNN", "Gride"],
                data=list(
                    zip(
                        layer_names,
                        mle_estimates,
                        mle_estimates_corr,
                        twonn_estimates,
                        gride_estimates,
                    )
                ),
            )
        else:
            print(
                f"Layer: {layer_name} || TwoNN: {twonn_estimates[-1]} || Gride: {gride_estimates[-1]}"
            )
            results = pd.DataFrame(
                columns=["Layer", "TwoNN", "Gride"],
                data=list(zip(layer_names, twonn_estimates, gride_estimates)),
            )

        results.to_csv(os.path.join(results_dir, "results.csv"), index=False)
        print(f"Deleting Data: {path}")

        os.system(f"rm -r {path}")

    os.system(f"rm -r {embeddings_dir}")
