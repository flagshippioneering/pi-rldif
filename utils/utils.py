import torch
from typing import *
import yaml
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="./transformers")

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

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(config)

def get_polynomial_learning_rate(
    optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    train_dataloader,
    step_scale_factor: float,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):

    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step: int):
        if num_training_steps is None:
            try:
                num_training_steps_final = len(train_dataloader) * step_scale_factor
            except TypeError:
                num_training_steps_final = step_scale_factor
            num_training_steps_final = int(num_training_steps_final)
        else:
            num_training_steps_final = num_training_steps

        # print("NUM TRAINING STEPS", num_training_steps_final)

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps_final:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps_final - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def featurize_GTrans(batch):
    """ Pack and pad batch into torch tensors """
    # alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    for i in range(len(batch)):
        chain_length = len(batch[i]['seq'])
        chain_mask = np.ones(chain_length)
        batch[i]['chain_mask'] =  chain_mask
        batch[i]['chain_encoding'] =  1*chain_mask
    batch = [one for one in batch if one is not None]
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])-1 
    chain_encoding = np.zeros([B, L_max])-1
    

    # Build the batch
    for i, b in enumerate(batch):
        #import pdb; pdb.set_trace()
        #x = b['bb']
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        # indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)


        S[i, :l] = indices
        chain_mask[i,:l] = b['chain_mask']
        chain_encoding[i,:l] = b['chain_encoding']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    chain_mask = torch.where(chain_mask == -1, torch.tensor(0), chain_mask)
    chain_encoding = torch.where(chain_encoding == -1, torch.tensor(0), chain_encoding)
    #
    #{"name": [b['name'] for b in batch],
    return {"name": [b['name'] for b in batch],
            "X":X,
            "S":S,
            "score": score,
            "mask":mask,
            "lengths":lengths,
            "chain_mask":chain_mask,
            "chain_encoding":chain_encoding}

def slice_dict(d: dict, keys: Iterable[Any], fail_if_missing: bool = False) -> dict:
    if fail_if_missing:
        return {k: d[k] for k in keys}
    else:
        if not isinstance(keys, set):
            # calling set(set(...)) does recompute the hash table, I think, based on some timing tests
            keys = set(keys)
        return {k: v for k, v in d.items() if k in keys}

mpnn_index_to_AA = {
    0: 'A',
    1: 'R',
    2: 'N',
    3: 'D',
    4: 'C',
    5: 'Q',
    6: 'E',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'L',
    11: 'K',
    12: 'M',
    13: 'F',
    14: 'P',
    15: 'S',
    16: 'T',
    17: 'W',
    18: 'Y',
    19: 'V',
    20: 'X'
}



