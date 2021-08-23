import numpy as np
import torch
import torch.nn as nn
from typing import List

############################
### Simplified version of https://github.com/cybertronai/autograd-hacks
############################

def _get_output(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    setattr(layer, "embeds", output.detach())


def _add_hook(model, layer):
    """
    Adds a forward hook that captures output of layer to model
    Args:
        model: pytorch model
        layer: String, name of a layer in model

    Returns: None
    """
    module = dict(model.named_modules())[layer]
    hook = module.register_forward_hook(_get_output)

    model.__dict__.setdefault("embed_hook", []).extend([hook])


def _remove_hooks(model):
    for handle in model.embed_hook:
        handle.remove()
    del model.embed_hook


def _read_hook(model, layer):
    module = dict(model.named_modules())[layer]
    embed = getattr(module, "embeds").detach().cpu()
    embed = embed.reshape(embed.shape[0], -1)
    return embed


def get_embeds(model, data, layer, gpu=0):
    """
    Returns the embeddings (outputs of a specified layer) in the model
    for all points in data
    Args:
        model: pytorch model
        data: DataLoader, ideally sequential
        layer: String, name of a layer in model

    Returns: a numpy array of size (n, e), where n is the number of observations in data
             and e is the size of the flattened embedding of the layer

    """
    assert layer in [names for names, mod in model.named_modules()], "{} is no layer of the model".format(layer)

    model.eval()
    _add_hook(model, layer)
    embeds = []
    for batch in data:
        if gpu >= 0:
            batch = batch.cuda()

        model(batch)
        embeds.append(_read_hook(model, layer))

    _remove_hooks(model)

    embeds = np.array(torch.cat(embeds))
    return embeds