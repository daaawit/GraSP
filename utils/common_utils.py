import os

import torch

from pprint import pprint

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def try_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def tensor_to_list(tensor):
    if len(tensor.shape) == 1:
        return [tensor[_].item() for _ in range(tensor.shape[0])]
    else:
        return [tensor_to_list(tensor[_]) for _ in range(tensor.shape[0])]


# =======================================================
# For math computation
# =======================================================
def prod(l):
    val = 1
    if isinstance(l, list):
        for v in l:
            val *= v
    else:
        val = val * l

    return val