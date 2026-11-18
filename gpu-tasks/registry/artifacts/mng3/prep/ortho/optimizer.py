"""Code for the ortho setup that train_gpt.py does not have: the definitions it adds, and those that replace
train_gpt.py's definitions of the same name (each marked). From modded-nanogpt track 3, submission 20260517_ortho,
at commit 4ea6b937. The constants are the ones its training run used."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# replaces train_gpt.py's muon_update
def muon_update(grad, momentum, mu=0.95):
    momentum.lerp_(grad, 1 - mu)
    update = zeropower_via_newtonschulz5(momentum)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update
