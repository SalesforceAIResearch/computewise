"""Code for the newton-muon setup that train_gpt.py does not have: the definitions it adds, and those that replace
train_gpt.py's definitions of the same name (each marked). From modded-nanogpt track 3, submission 20260505_newton_muon,
at commit 4ea6b937. The constants are the ones its training run used."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

def accum_xtx_(x: Tensor, accum: Tensor, count: Tensor):
    x = x.detach().flatten(0, -2).float()
    if accum.ndim == 2:
        accum.add_(x.T @ x, alpha=1 / x.size(0))
    else:
        n, d = x.size(0), accum.size(-1)
        x = x.reshape(n, 4, d).permute(1, 2, 0)
        accum.add_(torch.bmm(x, x.mT), alpha=1 / n)
    count.add_(1)


# replaces train_gpt.py's CausalSelfAttention
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        self.register_buffer("qkv_xtx", torch.zeros(dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("qkv_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("o_xtx", torch.zeros(hdim, hdim, dtype=torch.float32), persistent=False)
        self.register_buffer("o_count", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


# replaces train_gpt.py's MLP
class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.register_buffer("fc_xtx", torch.zeros(dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("fc_count", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("proj_xtx", torch.zeros(4, dim, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("proj_count", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


def attach_precond_stats(model: GPT):
    for block in model.blocks:
        a, m = block.attn, block.mlp
        qkv = dict(accum=a.qkv_xtx, count=a.qkv_count)
        for mod, ref in ((a.q, qkv), (a.k, qkv), (a.v, qkv),
                         (a.proj, dict(accum=a.o_xtx, count=a.o_count)),
                         (m.fc, dict(accum=m.fc_xtx, count=m.fc_count)),
                         (m.proj, dict(accum=m.proj_xtx, count=m.proj_count))):
            mod.weight._stats_ref = ref


def precond_stat_hook(module, inputs, output):
    ref = module.weight._stats_ref
    accum_xtx_(inputs[0], ref["accum"], ref["count"])


def enable_precond_hooks(model: GPT):
    return [m.register_forward_hook(precond_stat_hook)
            for b in model.blocks for m in (b.attn.q, b.attn.proj, b.mlp.fc, b.mlp.proj)]


# replaces train_gpt.py's Muon
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, refresh_interval=64):
        assert isinstance(params, list) and len(params) >= 1
        if isinstance(params[0], torch.nn.Parameter):
            params = sorted(params, key=lambda x: x.size(), reverse=True)
        else:
            assert isinstance(params[0], dict)
            params = [dict(group, params=sorted(list(group["params"]), key=lambda x: x.size(), reverse=True))
                      for group in params]
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self.global_step = 0
        self._precond_attached = self._precond_has_inv = False
        self._precond_stats = []
        self._precond_K = self._precond_count = None
        self._precond_cov = None
        self._precond_matrix_stat_idx = None
        self._precond_matrix_count = self._precond_matrix_count_clamped = None
        self._precond_alpha = None
        self.refresh_interval = refresh_interval

    @staticmethod
    def _eye_(x, value):
        x.zero_()
        x.diagonal(dim1=-2, dim2=-1).fill_(value)

    @torch.no_grad()
    def attach_preconditioner(self):
        self._precond_attached = True
        stats = {}
        for group in self.param_groups:
            for p in group["params"]:
                ref = getattr(p, "_stats_ref", None)
                assert ref is not None, f"Missing preconditioner stats for parameter {tuple(p.shape)}"
                key = id(ref["accum"])
                if key not in stats:
                    stats[key] = dict(accum=ref["accum"], count=ref["count"])
                self.state[p]["precond"] = stats[key]
        self._precond_stats = list(stats.values())
        d = self._precond_stats[0]["accum"].size(-1)
        n_by_stat = [s["accum"].numel() // (d * d) for s in self._precond_stats]
        n = sum(n_by_stat)
        device = self._precond_stats[0]["accum"].device
        self._precond_K = torch.empty(n, d, d, device=device)
        self._precond_cov = torch.empty_like(self._precond_K)
        self._precond_count = torch.empty(len(self._precond_stats), device=device)
        self._precond_matrix_count = torch.empty(n, device=device)
        self._precond_matrix_count_clamped = torch.empty(n, device=device)
        self._precond_matrix_stat_idx = torch.repeat_interleave(
            torch.arange(len(self._precond_stats), device=device),
            torch.tensor(n_by_stat, device=device),
        )
        self._precond_alpha = torch.empty(n, 1, 1, device=device)
        i = 0
        for s, stat_n in zip(self._precond_stats, n_by_stat):
            s["n"] = stat_n
            s["cov"] = self._precond_cov[i:i+stat_n].reshape_as(s["accum"])
            s["inv"] = self._precond_K[i:i+stat_n].reshape_as(s["accum"])
            self._eye_(s["cov"], 0.001)
            self._eye_(s["inv"], 1.0)
            s["accum"].zero_()
            s["count"].zero_()
            i += stat_n

    def precond_flag_for_step(self, step: int):
        return self._precond_attached and (int(step) + 1) % self.refresh_interval == 0

    @torch.no_grad()
    def _refresh_preconditioner(self):
        K, d, i = self._precond_K, self._precond_K.size(-1), 0
        for j, s in enumerate(self._precond_stats):
            n = s["n"]
            K[i:i+n].copy_(s["accum"].reshape(n, d, d))
            self._precond_count[j].copy_(s["count"])
            i += n
        dist.all_reduce(K, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._precond_count, op=dist.ReduceOp.SUM)

        torch.index_select(self._precond_count, 0, self._precond_matrix_stat_idx, out=self._precond_matrix_count)
        self._precond_matrix_count_clamped.copy_(self._precond_matrix_count).clamp_min_(1)
        K.div_(self._precond_matrix_count_clamped.view(-1, 1, 1))
        self._precond_alpha.copy_(self._precond_matrix_count.gt(0).view(-1, 1, 1)).mul_(0.05)
        self._precond_cov.lerp_(K, self._precond_alpha)

        diag = self._precond_cov.diagonal(dim1=-2, dim2=-1)
        reg = (diag.sum(-1) / d * 0.2 + 1e-8).unsqueeze(-1)
        diag.add_(reg)
        L, info = torch.linalg.cholesky_ex(self._precond_cov, upper=False, check_errors=False)
        diag.sub_(reg)
        torch.cholesky_inverse(L, upper=False, out=K)
        if info.any():
            self._eye_(K[info != 0], 1.0)

        for s in self._precond_stats:
            s["accum"].zero_()
            s["count"].zero_()
        self._precond_has_inv = True

    @torch.no_grad()
    def _precondition_grads(self):
        if not self._precond_has_inv:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad is not p.grad:
                    p.grad = grad
                inv = self.state[p]["precond"]["inv"]
                if inv.ndim == 3:
                    d = inv.size(-1)
                    g = grad.view(grad.size(0), 4, d).permute(1, 0, 2)
                    p.grad.copy_(torch.bmm(g, inv).permute(1, 0, 2).reshape_as(p.grad))
                else:
                    p.grad.copy_(grad @ inv)

    @torch.no_grad()
    def step(self):
        if self._precond_attached and self.precond_flag_for_step(self.global_step):
            self._refresh_preconditioner()
        if self._precond_attached:
            self._precondition_grads()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.global_step += 1
