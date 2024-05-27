import torch
from typing import *

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
LongType = Union[int, LongTensor]

TRAIN = 0
VALIDATION = 1
INFERENCE = 2
SAMPLING = 3

N_AA = 21

def t2n(x):
    """Convert torch tensor(s) to numpy array(s)

    Args:
        x (Any): Must be a tensor, a list of tensors, or a tensor-valued dict.
            Compositions of the above are allowed, for example a list of tensor-valued
            dicts, or a dict mapping strings to list of tensors. Anything that doesn't
            conform to the above (e.g. None or np.array) will be returned unchanged.

    Returns:
        Something numerical.
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if len(x.shape) == 0:
            return x.item()
        else:
            return x.numpy()
    elif isinstance(x, list):
        return [t2n(xx) for xx in x]
    elif isinstance(x, dict):
        return {k: t2n(v) for k, v in x.items()}
    else:
        return x
