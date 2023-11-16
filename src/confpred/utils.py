from typing import Optional, Tuple, Iterable

import numpy as np
import torch
from torch import nn
from torch import Generator, Tensor

class BufferList(nn.Module):
    """
    Stores a list of buffers, similar to nn.ParameterList
    """
    def __init__(self, buffers: Optional[Iterable[Tensor]] = None) -> None:
        super().__init__()
        self.extend(buffers)
    
    def extend(self, buffers: Iterable[Tensor]):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self
    
    def replace(self, buffers: Iterable[Tensor]):
        """
        Replaces data in with buffers (must match length of self._buffers)
        """
        for i, buffer in enumerate(buffers):
            self.get_buffer(str(i)).data = buffer
    
    def __len__(self):
        return len(self._buffers)
    
    def __iter__(self):
        return iter(self._buffers.values())
        

def soft_threshold(vals: Tensor, threshold: Tensor, alpha=0.2):
    return torch.sigmoid((vals - threshold)/alpha)


def split(num_splits, *tensors: Tuple[Tensor]) -> Tuple[Tuple[Tensor]]:
    batch_size = np.ceil(tensors[0].shape[0] / num_splits)
    return tuple(
        tuple(
            tensor[split * batch_size : (split + 1) * batch_size] for tensor in tensors
        )
        for split in range(num_splits)
    )


def random_split(
    num_splits, *tensors: Tuple[Tensor], generator: Optional[Generator] = None
) -> Tuple[Tuple[Tensor]]:
    batch_size = int(np.ceil(tensors[0].shape[0] / num_splits))
    randomized_order = torch.randperm(tensors[0].shape[0], generator=generator)
    return tuple(
        tuple(
            tensor[randomized_order[split * batch_size : (split + 1) * batch_size]]
            for tensor in tensors
        )
        for split in range(num_splits)
    )

def gaussian_kl_div(mu1, lnvar1, mu2, lnvar2, **kwargs):
    mu1 = mu1.flatten()
    lnvar1 = lnvar1.flatten()
    
    mu2 = mu2.flatten()
    lnvar2 = lnvar2.flatten()
    
    kl_div = 0.5 * (lnvar2 - lnvar1) # 1/2 log (Var2 / Var1)
    kl_div += 0.5 * torch.exp(lnvar1 - lnvar2) # Var1 / 2 Var2
    kl_div += 0.5 * (mu2 - mu1)**2 / torch.exp(lnvar2)
    kl_div -= 0.5
    
    return torch.sum(kl_div)