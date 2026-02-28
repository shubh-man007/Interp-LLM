import os
import torch
import numpy as np

def hook_layers(name):
    return (
        name.endswith("hook_mlp_out")
        or name.endswith("hook_resid_post")
        or name.endswith("ln_final.hook_normalized")
    )


def save_hook_function(model_dir, batch):
    def saving(activation, hook):
        activation = activation.detach().cpu()
        activation = activation.reshape(activation.shape[0], -1)
        os.path.join(model_dir, f"{hook.name}")
        os.makedirs(os.path.join(model_dir, f"{hook.name}"), exist_ok=True)
        torch.save(
            activation, os.path.join(model_dir, f"{hook.name}", f"batch_{batch}.pt")
        )
        if batch % 10 == 0:
            print(f"Saved {hook.name}| Batch: {batch}| Shape: {activation.shape}")

    return saving


def save_last_token_hook_function(model_dir, batch, idx):
    def saving(activation, hook):
        activation = activation.detach().cpu()
        activation = activation[torch.arange(activation.shape[0]), idx]
        os.path.join(model_dir, f"{hook.name}")
        os.makedirs(os.path.join(model_dir, f"{hook.name}"), exist_ok=True)
        torch.save(
            activation, os.path.join(model_dir, f"{hook.name}", f"batch_{batch}.pt")
        )
        if batch % 10 == 0:
            print(f"Saved {hook.name}| Batch: {batch}| Shape: {activation.shape}")

    return saving


def save_particular_token_hook_function(
    model_dir, batch, idx, relative_depth_of_sentence=1.0
):
    if torch.allclose(torch.tensor(relative_depth_of_sentence), torch.tensor(1.0)):
        idx = idx
    else:
        idx = torch.tensor(relative_depth_of_sentence * idx)
        idx = torch.round(idx).long()

    def saving(activation, hook):
        activation = activation.detach().cpu()
        activation = activation[torch.arange(activation.shape[0]), idx]
        activation = activation.reshape(activation.shape[0], -1)
        os.path.join(model_dir, f"{hook.name}")
        os.makedirs(os.path.join(model_dir, f"{hook.name}"), exist_ok=True)
        torch.save(
            activation, os.path.join(model_dir, f"{hook.name}", f"batch_{batch}.pt")
        )
        if batch % 10 == 0:
            print(f"Saved {hook.name}| Batch: {batch}| Shape: {activation.shape}")

    return saving


def name_all_hook_layers(model):
    for name, _ in model.named_modules():
        print(name)


def return_particular_token_hook_function(
    return_variable, batch, idx, relative_depth_of_sentence=1.0
):
    if torch.allclose(torch.tensor(relative_depth_of_sentence), torch.tensor(1.0)):
        idx = idx
    else:
        idx = torch.tensor(relative_depth_of_sentence * idx)
        idx = torch.round(idx).long()

    def return_activation(activation, hook):
        activation = activation[torch.arange(activation.shape[0]), idx]
        activation = activation.reshape(activation.shape[0], -1)
        return_variable[hook.name] = activation
        if batch % 10 == 0:
            print(f"Returned {hook.name}| Batch: {batch}| Shape: {activation.shape}")

    return return_activation
