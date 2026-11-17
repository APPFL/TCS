"""MNIST reproduction resources (two hidden layers, raw [0, 1] pixels)."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST


class MNISTMLP(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )


def partition_labels(labels, num_clients, alpha, seed):
    """Class-wise Dirichlet allocation; repair empty clients with donor samples.

    Repair takes one random sample from the currently largest client. No sample
    is duplicated or discarded; ties use client order. Persist the partition hash.
    """
    if not 0 < num_clients <= len(labels) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid client count or Dirichlet alpha")
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        counts = rng.multinomial(
            len(indices), rng.dirichlet(np.full(num_clients, alpha))
        )
        for group, block in zip(groups, np.split(indices, np.cumsum(counts)[:-1])):
            group.extend(block.tolist())
    for group in groups:
        if not group:
            donor = max(groups, key=len)
            group.append(donor.pop(int(rng.integers(len(donor)))))
    return groups


def load_mnist(root, num_clients=500, alpha=0.1, seed=42):
    raw = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    train_data = TensorDataset(raw.data.unsqueeze(1).float().div_(255), raw.targets)
    test_data = TensorDataset(test.data.unsqueeze(1).float().div_(255), test.targets)
    groups = partition_labels(raw.targets.numpy(), num_clients, alpha, seed)
    return train_data, test_data, groups


def get_dataset(root="datasets", num_clients=2, client_id=0, alpha=0.1, seed=42):
    train, test, groups = load_mnist(root, num_clients, alpha, seed)
    return Subset(train, groups[client_id]), test


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(str((key, tuple(tensor.shape), str(tensor.dtype))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, dataset, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    correct, loss, count = 0, 0.0, 0
    for x, y in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return dict(accuracy=100 * correct / count, loss=loss / count, count=count)
