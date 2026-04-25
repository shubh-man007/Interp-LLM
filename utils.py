import gc
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from custom_dataloaders.mmlu import MMLUDataset
from custom_dataloaders.cola import ColaDataset
from custom_dataloaders.sst2 import SST2Dataset
from custom_dataloaders.rotten_tomatoes import RottenTomatoesDataset
from custom_dataloaders.agnews import AGNewsDataset
from custom_dataloaders.arithmetic import ArithmeticDataset
from custom_dataloaders.year import YearDataset
from custom_dataloaders.copa import CopaDataset
from custom_dataloaders.cold import ColdDataset

def getDataset(args, dataset_name, tokenizer, detokenizer):
    if dataset_name.split("_")[0] == "mmlu":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"

        dataset = MMLUDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "cola":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "train"

        dataset = ColaDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "sst2":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "train"

        dataset = SST2Dataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "rottentomatoes":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"

        dataset = RottenTomatoesDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "agnews":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"

        dataset = AGNewsDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "arithmetic":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"
        dataset = ArithmeticDataset(
            tokenizer,
            detokenizer,
            args.max_token_length,
            "right",
            "validation",
            args.last_token,
            # args.question_structure,
            False,
            args.few_shot,
            seed=args.random_seed,
        )
    elif dataset_name.split("_")[0] == "year":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"
        dataset = YearDataset(
            tokenizer,
            detokenizer,
            # years_to_sample_from=[1960, 2021],
            N=2000,
            nouns="war",
            max_token_length=args.max_token_length,
            padding_side="right",
            split=split,
            last_token=args.last_token,
            add_question_structure=False,
            icl_few_shot=args.few_shot,
            balanced=True,
            eos=False,
            device=args.device,
            seed=args.random_seed,
        )
    elif dataset_name.split("_")[0] == "copa":
        if "_" in dataset_name:
            split = "_".join(dataset_name.split("_")[1:])
        else:
            split = "test"
        dataset = CopaDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            split=split,
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
        )
    elif dataset_name.split("_")[0] == "cold":
        if "_" in dataset_name:
            activity = " ".join(dataset_name.split("_")[1:])
        else:
            activity = "going grocery shopping"
        dataset = ColdDataset(
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            max_token_length=args.max_token_length,
            padding_side="right",
            split="train",
            last_token=args.last_token,
            add_question_structure=args.question_structure,
            icl_few_shot=args.few_shot,
            activity=activity,
            num_samples=1000,
            seed=args.random_seed,
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return dataset


def getSaveDirectories(args):
    model_id = args.model_id.replace("/", "_")
    if args.pythia:
        model_id += f"_step{args.revision_step}"
    dataset_name = args.dataset_name
    few_shot = args.few_shot
    relative_depth_of_sentence = args.relative_depth_of_sentence

    embeddings_dir = os.path.join(
        "models_hook",
        dataset_name,
        f"few_shot_{few_shot}",
        model_id,
    )
    results_dir = os.path.join(
        # "results",
        args.save_dir,
        dataset_name,
        f"few_shot_{few_shot}",
        model_id,
    )
    if args.last_token:
        embeddings_dir = os.path.join(
            embeddings_dir,
            f"last_token_relative_depth_{round(relative_depth_of_sentence * 100)}",
        )
        results_dir = os.path.join(
            results_dir,
            f"last_token_relative_depth_{round(relative_depth_of_sentence * 100)}",
        )
    else:
        embeddings_dir = os.path.join(embeddings_dir, "all_tokens")
        results_dir = os.path.join(results_dir, "all_tokens")

    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    accuracy_save_path = os.path.join(results_dir, f"accuracy.json")
    label_save_path = os.path.join(results_dir, f"predictions.csv")
    return embeddings_dir, label_save_path, accuracy_save_path, results_dir
