"""Code for the soda-fade setup that train_gpt.py does not have: the definitions it adds, and those that replace
train_gpt.py's definitions of the same name (each marked). From modded-nanogpt track 3, submission 20260518_soda_fade_3000_n9,
at commit 4ea6b937. The constants are the ones its training run used."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import math

SODA_HIDDEN = True
SODA_LAMBDA_SCALE = 1.0
SODA_FADE_START_STEP = 2000
SODA_FADE_END_STEP = 2750
SODA_FADE_SHAPE = "cosine"
SEED = args.seed
FINAL_TRAIN_STEPS = 3040
FINAL_SCHEDULE_STEPS = 3105
FINAL_LR_POWER = 1.2
ADAM_EMBED_POWER_C = 4.976805410800738e-05
ADAM_PROJ_POWER_C = 5.184172302917436e-07
ADAM_OTHER_POWER_C = 1.6589351369335795e-06
MUON_POWER_C = 3.3169534699576625e-06
FINAL_MUON_WD = 0.025
EXPERIMENT_NAME = "contra-muon-to-soft-muon"
EXPERIMENT_INTUITION = "Interpolate Muon update from Contra Muon to normal Muon, then from normal Muon to p=0.1 soft Muon."
CONTRA_MUON_COEFF = -0.2
SOFT_MUON_P = 0.1
SOFT_MUON_SCALE = "none"
SOFT_MUON_INPUT_NORM = "frobenius_schatten4"
SOFT_MUON_CEIL = 0.80
CONTRA_HOLD_END_STEP = 0
CONTRA_TO_NORMAL_END_STEP = 2000
NORMAL_TO_SOFT_START_STEP = 2500
NORMAL_TO_SOFT_END_STEP = 3010
MU = 0.95
MUON_LR = 0.0375
MUON_WEIGHT_DECAY = FINAL_MUON_WD
TARGET_UW = 0.3825
SOAP_TARGET_UW = TARGET_UW
NONSOAP_TARGET_UW = TARGET_UW
NOR_BETA2 = 0.95
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 10
SOAP_DENOM_POWER = 0.50
SOAP_BLEND = 1.00
SOAP_UPDATE_BEFORE_USE = False
SOAP_PARAM_MODE = "mlp_plus_v"
ATTN_SOAP_DENOM_FLOOR = 0.55
ATTN_SOAP_BLEND = 1.00
V_SOAP_BLEND = 0.95
V_SOAP_BLEND_RAMP_END_STEP = 0
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_STEP = 1375
ATTN_TRUST_FLOOR_FADE_END_STEP = 1625
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
ATTN_SOAP_FADE_START_STEP = 1000000000
ATTN_SOAP_FADE_END_STEP = 1000000000
NO_CONTRA_PARAM = ""
NO_SOFTMUON_PARAM = ""
TRAIN_PROGRESS_INTERVAL = 0
LOG_DIR = Path("experiment-logs/all-runs/2026-05-09/validation-ceil-0.8")


def soda_fade_multiplier(step: int) -> float:
    if step < SODA_FADE_START_STEP:
        return 1.0
    if step >= SODA_FADE_END_STEP:
        return 0.0
    x = (step - SODA_FADE_START_STEP) / (SODA_FADE_END_STEP - SODA_FADE_START_STEP)
    if SODA_FADE_SHAPE == "linear":
        return 1.0 - x
    return 0.5 * (1.0 + math.cos(math.pi * x))


# replaces train_gpt.py's distributed_data_generator
def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pattern = Path(filename_pattern)
    files = sorted(pattern.parent.glob(pattern.name))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


# replaces train_gpt.py's RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)


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

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


def gram_frobenius_norm_estimate(G: Tensor, keepdim: bool = False, eps: float = 1e-10) -> Tensor:
    X = G.float()
    gram = X.mT @ X if X.size(-2) > X.size(-1) else X @ X.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)


# replaces train_gpt.py's zeropower_via_newtonschulz5
def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Use the Gram-Frobenius estimate sqrt(||X X^T||_F) instead of direct
    # Frobenius norm or power iteration.
    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _soft_coefficients(p: float) -> tuple[float, tuple[float, ...]]:
    if p == 0.0:
        return 1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if p == 0.1:
        return 0.0, (0.1091613623, 0.07085664498, 0.05210528973, 0.05457295795, 0.05011334061, 0.03334622198, 0.05022104481, 0.1053727358, 0.1187323776, 0.1185061091, 0.1185059576, 0.1185059576)
    raise ValueError(f"unsupported soft-muon singular-value power: {p}")


def zeropower_frobenius_norm_like(G: Tensor) -> float:
    return (G.numel() / max(G.size(-2), G.size(-1)))**0.5


def soft_via_newtonschulz5(G: Tensor, p: float, scale_mode: str, input_norm: str) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    if input_norm == "frobenius_schatten4":
        X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    elif input_norm != "frobenius":
        raise ValueError(f"unsupported soft_muon input_norm: {input_norm}")
    else:
        X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    constant, coeffs = _soft_coefficients(p)

    a, b, c = 2, -1.5, 0.5
    basis = [X]
    for _ in range(len(coeffs)):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        basis.append(X)

    out = constant * basis[-1]
    for coeff, basis_term in zip(coeffs, basis[:-1]):
        out = out + coeff * basis_term

    value_at_one = constant + sum(coeffs)
    if scale_mode == "top":
        out = out / value_at_one
    elif scale_mode == "top_sqrt":
        out = out / value_at_one**0.5
    elif scale_mode == "frobenius":
        theoretical_opower_norm = zeropower_frobenius_norm_like(out)
        out = out * theoretical_opower_norm / gram_frobenius_norm_estimate(out)
    elif scale_mode == "frobenius_sqrt":
        theoretical_opower_norm = zeropower_frobenius_norm_like(out)
        out = out * (theoretical_opower_norm / gram_frobenius_norm_estimate(out))**0.5
    elif scale_mode != "none":
        raise ValueError(f"unsupported soft_muon scale: {scale_mode}")

    if G.size(-2) > G.size(-1):
        out = out.mT
    return out


def scale_to_unit_operator_norm(G: Tensor, eps: float = 1e-10) -> Tensor:
    return G / gram_frobenius_norm_estimate(G, eps=eps).to(G.dtype)


def should_soap_param(name: str) -> bool:
    is_mlp_fc = name.endswith(".mlp.fc.weight")
    is_mlp_proj = name.endswith(".mlp.proj.weight")
    is_attn_proj = name.endswith(".attn.proj.weight")
    is_qkv = (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
    )
    is_q = name.endswith(".attn.q.weight")
    is_k = name.endswith(".attn.k.weight")
    is_v = name.endswith(".attn.v.weight")
    if SOAP_PARAM_MODE == "mlp_all":
        return is_mlp_fc or is_mlp_proj
    if SOAP_PARAM_MODE == "mlp_fc":
        return is_mlp_fc
    if SOAP_PARAM_MODE == "mlp_proj":
        return is_mlp_proj
    if SOAP_PARAM_MODE == "mlp_plus_attn_proj":
        return is_mlp_fc or is_mlp_proj or is_attn_proj
    if SOAP_PARAM_MODE == "mlp_plus_q":
        return is_mlp_fc or is_mlp_proj or is_q
    if SOAP_PARAM_MODE == "mlp_plus_k":
        return is_mlp_fc or is_mlp_proj or is_k
    if SOAP_PARAM_MODE == "mlp_plus_v":
        return is_mlp_fc or is_mlp_proj or is_v
    if SOAP_PARAM_MODE == "mlp_plus_qkv":
        return is_mlp_fc or is_mlp_proj or is_qkv
    if SOAP_PARAM_MODE == "all_hidden":
        return is_mlp_fc or is_mlp_proj or is_attn_proj or is_qkv
    raise ValueError(f"unknown SOAP_PARAM_MODE={SOAP_PARAM_MODE}")


def is_attn_proj_param(name: str) -> bool:
    return name.endswith(".attn.proj.weight")


def is_attn_param(name: str) -> bool:
    return (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
        or name.endswith(".attn.proj.weight")
    )


def is_v_param(name: str) -> bool:
    return name.endswith(".attn.v.weight")


def param_matches_spec(name: str, spec: str) -> bool:
    keys = {part.strip() for part in spec.split(",") if part.strip()}
    return (
        ("q" in keys and name.endswith(".attn.q.weight"))
        or ("k" in keys and name.endswith(".attn.k.weight"))
        or ("v" in keys and name.endswith(".attn.v.weight"))
        or ("attn_proj" in keys and name.endswith(".attn.proj.weight"))
    )


def tensor_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_f, b_f = a.float(), b.float()
    return (a_f * b_f).sum() / (a_f.norm() * b_f.norm()).clamp_min(eps)


def trust_gate(raw: Tensor, soap: Tensor, grad: Tensor, eps: float = 1e-8) -> Tensor:
    # SOAP is trusted when it still points with raw momentum and is at least as
    # gradient-aligned as raw momentum. This catches stale whitening bases.
    raw_grad = tensor_cosine(raw, grad, eps)
    soap_grad = tensor_cosine(soap, grad, eps)
    soap_raw = tensor_cosine(soap, raw, eps)

    agree_gate = ((soap_raw - ATTN_TRUST_MIN_AGREE) / (1 - ATTN_TRUST_MIN_AGREE)).clamp(0, 1)
    denom = (raw_grad - ATTN_TRUST_MIN_GRAD_ALIGN).clamp_min(eps)
    grad_gate = ((soap_grad - ATTN_TRUST_MIN_GRAD_ALIGN) / denom).clamp(0, 1)
    gate = (agree_gate * grad_gate).clamp(0, 1)
    if ATTN_TRUST_POWER != 1.0:
        gate = gate.pow(ATTN_TRUST_POWER)
    return gate


def early_trust_floor_for_step(step: int) -> float:
    if ATTN_TRUST_FLOOR_FADE_END_STEP <= ATTN_TRUST_FLOOR_END_STEP:
        return 0.0 if step >= ATTN_TRUST_FLOOR_FADE_END_STEP else ATTN_EARLY_TRUST_FLOOR
    if step < ATTN_TRUST_FLOOR_END_STEP:
        return ATTN_EARLY_TRUST_FLOOR
    if step >= ATTN_TRUST_FLOOR_FADE_END_STEP:
        return 0.0
    return ATTN_EARLY_TRUST_FLOOR * (
        ATTN_TRUST_FLOOR_FADE_END_STEP - step
    ) / (ATTN_TRUST_FLOOR_FADE_END_STEP - ATTN_TRUST_FLOOR_END_STEP)


def bounded_trust_gate(gate: Tensor, step: int) -> Tensor:
    floor = early_trust_floor_for_step(step)
    cap = ATTN_EARLY_TRUST_CAP if step < ATTN_TRUST_FLOOR_FADE_END_STEP else 1.0
    return gate.clamp(min=floor, max=cap)


def attention_soap_blend_for_step(step: int) -> float:
    if step < ATTN_SOAP_FADE_START_STEP:
        return 1.0
    if step >= ATTN_SOAP_FADE_END_STEP:
        return 0.0
    if ATTN_SOAP_FADE_END_STEP <= ATTN_SOAP_FADE_START_STEP:
        return 0.0
    return (
        ATTN_SOAP_FADE_END_STEP - step
    ) / (ATTN_SOAP_FADE_END_STEP - ATTN_SOAP_FADE_START_STEP)


def norm_preserving_blend(raw: Tensor, soap: Tensor, gate: Tensor, eps: float = 1e-8) -> Tensor:
    blended = raw + (soap - raw) * gate.to(raw.dtype)
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    blended_norm = gram_frobenius_norm_estimate(blended, eps=eps)
    return (blended * (raw_norm / blended_norm).to(blended.dtype)).to(raw.dtype)


def soap_eigenbasis(mat: Tensor) -> Tensor:
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.size(0), device=mat.device))
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * torch.eye(mat.size(0), device=mat.device))
        q = q.float()
    return torch.flip(q, [1])


def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)

    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq


def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8,
                               blend=SOAP_BLEND, denom_floor_ratio=0.0):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(SOAP_DENOM_POWER)
    if denom_floor_ratio > 0:
        denom_floor = denom.float().square().mean().sqrt().mul(denom_floor_ratio).clamp_min(eps)
        denom = denom.clamp_min(denom_floor.to(denom.dtype))
    precond = q_row @ (projected / denom) @ q_col.T
    if blend != 1.0:
        precond = blend * precond + (1 - blend) * update_f
    precond.mul_(gram_frobenius_norm_estimate(update_f, eps=eps) / gram_frobenius_norm_estimate(precond, eps=eps))
    return precond.to(update.dtype)


def soap_update_preconditioner(grad, state, shampoo_beta=SOAP_BETA2, precondition_frequency=SOAP_PRECONDITION_FREQUENCY):
    grad_f = grad.float()
    state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - shampoo_beta)
    state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - shampoo_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % precondition_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"]
        )
    state["soap_step"] += 1


def _linear_ramp(step: int, start_step: int, end_step: int) -> float:
    if end_step <= start_step:
        return 1.0 if step >= end_step else 0.0
    return min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))


def contra_coeff_for_step(step: int) -> float:
    contra_to_normal = _linear_ramp(step, CONTRA_HOLD_END_STEP, CONTRA_TO_NORMAL_END_STEP)
    return CONTRA_MUON_COEFF * (1.0 - contra_to_normal)


def soft_blend_for_step(step: int) -> float:
    return min(SOFT_MUON_CEIL, _linear_ramp(step, NORMAL_TO_SOFT_START_STEP, NORMAL_TO_SOFT_END_STEP))


# replaces train_gpt.py's muon_update
def muon_update(update, second_moment, step, beta2=NOR_BETA2, use_contra=True, use_soft=True):
    normalized_grad = scale_to_unit_operator_norm(update.clone())
    ns_update = zeropower_via_newtonschulz5(update)
    update_norm_estimate = gram_frobenius_norm_estimate(ns_update)

    contra_coeff = contra_coeff_for_step(step) if use_contra else 0.0
    contra_update = ns_update + contra_coeff * normalized_grad
    contra_update = contra_update * update_norm_estimate / gram_frobenius_norm_estimate(contra_update)
    if use_soft:
        soft_update = soft_via_newtonschulz5(update, SOFT_MUON_P, SOFT_MUON_SCALE, SOFT_MUON_INPUT_NORM)
        soft_update = soft_update * update_norm_estimate / gram_frobenius_norm_estimate(soft_update)
        blend = soft_blend_for_step(step)
    else:
        soft_update = contra_update
        blend = 0.0
    update = contra_update + (soft_update - contra_update) * blend
    update = update * update_norm_estimate / gram_frobenius_norm_estimate(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    # Per-row variance (or per-col if matrix is wide). Keep along the longer dim.
    if update.size(-2) >= update.size(-1):
        per_row_var = (update * update).mean(dim=-1, keepdim=True)  # shape (..., m, 1)
    else:
        per_row_var = (update * update).mean(dim=-2, keepdim=True)  # shape (..., 1, n)
    second_moment.lerp_(per_row_var.float(), 1 - beta2)
    vnorm = gram_frobenius_norm_estimate(update)
    update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
    vnorm_new = gram_frobenius_norm_estimate(update)
    update = update * (vnorm / vnorm_new)
    return update


# replaces train_gpt.py's Muon
class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.attn_proj_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_proj_param(n)}
        self.v_params = {p for n, p in named_params if is_v_param(n)}
        self.no_contra_params = {p for n, p in named_params if param_matches_spec(n, NO_CONTRA_PARAM)}
        self.no_soft_params = {p for n, p in named_params if param_matches_spec(n, NO_SOFTMUON_PARAM)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        if p in self.soap_params:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
                            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
                            state["q_row"] = None
                            state["q_col"] = None
                            state["soap_step"] = 0
                        # Second-moment buffer matches per-row-or-col shape.
                        if p.size(-2) >= p.size(-1):
                            state["second_moment"] = torch.zeros((*p.shape[:-1], 1),
                                dtype=torch.float32, device=p.device)
                        else:
                            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]),
                                dtype=torch.float32, device=p.device)
                        if SODA_HIDDEN:
                            state["soda_anchor"] = p.detach().clone()
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    is_attn_soap = p in self.attn_soap_params
                    use_soap = p in self.soap_params
                    if use_soap and SOAP_UPDATE_BEFORE_USE:
                        soap_update_preconditioner(grad, state)
                    if use_soap:
                        if is_attn_soap:
                            soap_blend = V_SOAP_BLEND if p in self.v_params else ATTN_SOAP_BLEND
                            if p in self.v_params and V_SOAP_BLEND_RAMP_END_STEP > 0:
                                soap_blend *= _linear_ramp(self.step_count, 0, V_SOAP_BLEND_RAMP_END_STEP)
                            soap_update = soap_precondition_momentum(
                                momentum_update, state, blend=soap_blend,
                                denom_floor_ratio=ATTN_SOAP_DENOM_FLOOR
                            )
                            if p in self.attn_proj_soap_params:
                                gate = bounded_trust_gate(
                                    trust_gate(momentum_update, soap_update, grad),
                                    self.step_count
                                )
                            else:
                                gate = torch.ones((), dtype=torch.float32, device=p.device)
                            gate = gate * attention_soap_blend_for_step(self.step_count)
                            momentum_update = norm_preserving_blend(momentum_update, soap_update, gate)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state, blend=SOAP_BLEND)
                    update = muon_update(
                        momentum_update,
                        state["second_moment"],
                        self.step_count,
                        use_contra=p not in self.no_contra_params,
                        use_soft=p not in self.no_soft_params,
                    )
                    # u/w-floor. SOAP and non-SOAP params can use different floors.
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    cur_uw = u_fro / p_fro
                    target_uw = SOAP_TARGET_UW if use_soap else NONSOAP_TARGET_UW
                    scale = torch.where(cur_uw < target_uw, target_uw * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    # WD set to 0 — u/w target replaces wd's role (smaller updates as p grows).
                    if SODA_HIDDEN:
                        soda_scale = SODA_LAMBDA_SCALE * soda_fade_multiplier(self.step_count)
                        lam = min(max(soda_scale / (self.step_count + 2), 0.0), 1.0)
                        p.mul_(1 - lam).add_(state["soda_anchor"], alpha=lam)
                    p.add_(update, alpha=-group["lr"])
                    if use_soap and not SOAP_UPDATE_BEFORE_USE:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1


# replaces train_gpt.py's print0
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)


def _lr(step, initial_lr, power_c, power=1.0):
    t_end = FINAL_SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)


def set_hparams(step):
    progress = step / FINAL_SCHEDULE_STEPS
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)
