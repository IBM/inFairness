import torch.nn


def freeze_network(network: torch.nn.Module):
    """Freeze network parameters.
    :param network: torch.nn.Module
    :type network: torch.nn.Module
    """
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network: torch.nn.Module):
    """Unfreeze network parameters.
    :param network: torch.nn.Module
    :type network: torch.nn.Module
    """
    for p in network.parameters():
        p.requires_grad = True
