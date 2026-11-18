"""Code for the aurora-ema-ref setup that train_gpt.py does not have: the definitions it adds, and those that replace
train_gpt.py's definitions of the same name (each marked). From modded-nanogpt track 3, submission 20260525_aurora_ema_ref,
at commit 4ea6b937. The constants are the ones its training run used."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse

SEED = args.seed
LOG_DIR = args.log_dir
LOG_FILE = args.log_file
TRAIN_PROGRESS_INTERVAL = args.progress_interval
TRAIN_STEPS = 2900
RUN_STOP_STEP = 2860
SCHEDULE_STEPS = 2980
LR_POWER = 1.2
REF_CAPTURE_STEP = 2375
REF_APPLY_START_STEP = 2850
REF_GAMMA = -0.075
ADAM_EMBED_POWER_C = 4.976805410800738e-05
ADAM_PROJ_POWER_C = 5.184172302917436e-07
ADAM_OTHER_POWER_C = 1.6589351369335795e-06
MUON_POWER_C = 3.3169534699576625e-06
CONTRA_MUON_COEFF = -0.2
CONTRA_HOLD_END_STEP = 0
CONTRA_TO_NORMAL_END_STEP = 2500
MU = 0.95
MUON_LR = 0.0375
MUON_COOLDOWN_STEPS = 100
TARGET_UW = 0.3825
NOR_BETA2 = 1.0
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 10
SOAP_DENOM_POWER = 0.50
SOAP_BLEND = 1.00
V_SOAP_BLEND = 0.95
ATTN_SOAP_DENOM_FLOOR = 0.55
RADIAL_OUTWARD_SCALE = 0.5
RADIAL_INWARD_SCALE = 1.0
CGI_ALPHA = 0.0
ZERO_NONPROJ_BIASES = True
DI_FC_ALPHA = 0.30
EMA_NESTEROV_STEPSIZE = 0.3
EMA_NESTEROV_LOOKAHEAD_EMA = 0.99
EMA_NESTEROV_PREFILL_STEPS = 300
EMA_NESTEROV_REST_STEP = TRAIN_STEPS - 950
_AURORA_K = 3
_AURORA_BETA = 0.25
_AURORA_EPS = 1e-7
_NUM_BLOCKS = len(model.blocks)
_MU_MIN = 0.85
_MU_MAX = 0.95
_MU_WARMUP_STEPS = 300
_MU_COOLDOWN_STEPS = MUON_COOLDOWN_STEPS


# replaces train_gpt.py's distributed_data_generator
def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
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


def _ns_inner(X: Tensor) -> Tensor:
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


# replaces train_gpt.py's zeropower_via_newtonschulz5
def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    is_originally_wide = G.size(-2) < G.size(-1)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    if is_originally_wide:
        Xt = X.mT
        Xt32 = Xt.to(torch.float32)
        target_row_sq = Xt.size(-1) / Xt.size(-2)
        row_norm = Xt32.norm(dim=-1, keepdim=True).clamp_(min=_AURORA_EPS)
        D = 1.0 / row_norm
        U = None
        for k in range(_AURORA_K):
            scaled = (D * Xt32).to(Xt.dtype)
            scaled_wide = scaled.mT
            scaled_wide = scaled_wide / gram_frobenius_norm_estimate(scaled_wide, keepdim=True, eps=1e-7).to(scaled_wide.dtype)
            U_wide = _ns_inner(scaled_wide)
            U = U_wide.mT
            if k < _AURORA_K - 1:
                U32 = U.to(torch.float32)
                row_sq = U32.pow(2).sum(dim=-1, keepdim=True).clamp_(min=_AURORA_EPS * _AURORA_EPS)
                D = D * (target_row_sq / row_sq).pow(_AURORA_BETA)
        X = U.mT
    else:
        X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
        X = _ns_inner(X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def scale_to_unit_operator_norm(G: Tensor, eps: float = 1e-10) -> Tensor:
    return G / gram_frobenius_norm_estimate(G, eps=eps).to(G.dtype)


def should_soap_param(name: str) -> bool:
    return (
        name.endswith(".mlp.fc.weight")
        or name.endswith(".mlp.proj.weight")
        or name.endswith(".attn.v.weight")
    )


def is_v_param(name: str) -> bool:
    return name.endswith(".attn.v.weight")


def is_attn_param(name: str) -> bool:
    return is_v_param(name)


def scale_radial_update(update: Tensor, param: Tensor, step: int, eps: float = 1e-12) -> Tensor:
    update_f = update.float()
    param_f = param.float()
    denom = (param_f * param_f).sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    # p.add_(update, alpha=-lr), so actual movement is -update.
    # Outward movement means (-update) is aligned with p, i.e. coeff < 0.
    outward_scale = RADIAL_OUTWARD_SCALE
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(outward_scale),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)


def target_radius_after_update(param: Tensor, update: Tensor, lr: float, eps: float = 1e-8) -> Tensor:
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    # Use only the radial component's first-order radius change as the intended
    # radius change; the post-step rescale below removes finite tangent drift.
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)


def rescale_to_radius(param: Tensor, target_norm: Tensor, eps: float = 1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))


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


def norm_preserving_soap_blend(raw: Tensor, soap: Tensor, eps: float = 1e-8) -> Tensor:
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    soap_norm = gram_frobenius_norm_estimate(soap, eps=eps)
    return (soap * (raw_norm / soap_norm).to(soap.dtype)).to(raw.dtype)


def _linear_ramp(step: int, start_step: int, end_step: int) -> float:
    if end_step <= start_step:
        return 1.0 if step >= end_step else 0.0
    return min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))


def contra_coeff_for_step(step: int) -> float:
    contra_to_normal = _linear_ramp(step, CONTRA_HOLD_END_STEP, CONTRA_TO_NORMAL_END_STEP)
    return CONTRA_MUON_COEFF * (1.0 - contra_to_normal)


# replaces train_gpt.py's muon_update
def muon_update(update, second_moment, step, beta2=NOR_BETA2):
    normalized_grad = scale_to_unit_operator_norm(update.clone())
    ns_update = zeropower_via_newtonschulz5(update)
    update_norm_estimate = gram_frobenius_norm_estimate(ns_update)
    contra_coeff = contra_coeff_for_step(step)
    update = ns_update + contra_coeff * normalized_grad
    update = update * update_norm_estimate / gram_frobenius_norm_estimate(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5

    if update.size(-2) >= update.size(-1):
        per_row_var = (update * update).mean(dim=-1, keepdim=True)
    else:
        per_row_var = (update * update).mean(dim=-2, keepdim=True)
    second_moment.lerp_(per_row_var.float(), 1 - beta2)
    vnorm = gram_frobenius_norm_estimate(update)
    update = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
    vnorm_new = gram_frobenius_norm_estimate(update)
    return update * (vnorm / vnorm_new)


# replaces train_gpt.py's Muon
class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.param_names = {p: n for n, p in named_params}
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.v_params = {p for n, p in named_params if is_v_param(n)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, mu=mu))

    def _init_state(self, p):
        state = self.state[p]
        if state:
            return state
        state["momentum"] = torch.zeros_like(p)
        if p in self.soap_params:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
            state["q_row"] = None
            state["q_col"] = None
            state["soap_step"] = 0
        if p.size(-2) >= p.size(-1):
            state["second_moment"] = torch.zeros((*p.shape[:-1], 1), dtype=torch.float32, device=p.device)
        else:
            state["second_moment"] = torch.zeros((*p.shape[:-2], 1, p.shape[-1]), dtype=torch.float32, device=p.device)
        return state

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
                    state = self._init_state(p)
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    use_soap = p in self.soap_params
                    if use_soap:
                        if p in self.attn_soap_params:
                            soap_update = soap_precondition_momentum(
                                momentum_update,
                                state,
                                blend=V_SOAP_BLEND,
                                denom_floor_ratio=ATTN_SOAP_DENOM_FLOOR,
                            )
                            momentum_update = norm_preserving_soap_blend(momentum_update, soap_update)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state, blend=SOAP_BLEND)
                    update = muon_update(momentum_update, state["second_moment"], self.step_count)
                    update = scale_radial_update(update, p, self.step_count)
                    p_fro = p.float().norm().clamp_min(1e-8)
                    u_fro = update.float().norm().clamp_min(1e-8)
                    scale = torch.where(u_fro / p_fro < TARGET_UW, TARGET_UW * p_fro / u_fro, torch.ones_like(p_fro))
                    update = update * scale.to(update.dtype)
                    target_radius = target_radius_after_update(p, update, group["lr"])
                    p.add_(update, alpha=-group["lr"])
                    rescale_to_radius(p, target_radius)
                    if use_soap:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1


class EMA_Nesterov(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        inner_optimizer,
        lookahead_stepsize=0.0,
        use_scheduled_lookahead_stepsize=True,
        lookahead_ema=0.9,
        prefill_steps=0,
        rest_steps=0,
    ):
        if lookahead_stepsize < 0.0:
            raise ValueError(f"invalid lookahead_stepsize: {lookahead_stepsize}")

        super().__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.prefill_steps = prefill_steps
        self.rest_steps = rest_steps
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0.0
        self.initialize_buffers()

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "prev_params" not in param_state:
                    param_state["prev_params"] = (p.clone(), self.it)
                if "lookahead_buffer" not in param_state:
                    param_state["lookahead_buffer"] = (torch.zeros_like(p), -1)

    def get_lr_lambda(self):
        inner = self.inner_optimizer[0] if isinstance(self.inner_optimizer, list) else self.inner_optimizer
        return inner.param_groups[0]["lr"] / inner.param_groups[0]["initial_lr"]

    @torch.no_grad()
    def lookahead_step(self):
        if self.use_scheduled_lookahead_stepsize:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda()
        else:
            lookahead_stepsize = self.lookahead_stepsize
        self.current_lookahead_stepsize = lookahead_stepsize

        for group in self.param_groups:
            for p in group["params"]:
                lookahead = self.state[p]["lookahead_buffer"][0]
                p.add_(lookahead, alpha=lookahead_stepsize)

    @torch.no_grad()
    def accum_lookahead(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                look = p.add(param_state["prev_params"][0], alpha=-1)
                buf = param_state["lookahead_buffer"][0]
                param_state["lookahead_buffer"] = (
                    buf.lerp_(look, 1 - self.lookahead_ema),
                    self.it,
                )
                param_state["prev_params"] = (param_state["prev_params"][0].copy_(p), self.it)

    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.prefill_steps and self.it < self.rest_steps and not self.lookahead_status:
            self.lookahead_step()
        else:
            self.current_lookahead_stepsize = 0.0
        self.lookahead_status = True

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            raise ValueError("optimizer.nesterov_step() should be invoked before model forward pass")
        if isinstance(self.inner_optimizer, list):
            for opt in self.inner_optimizer:
                opt.step()
        else:
            self.inner_optimizer.step()
        self.accum_lookahead()
        self.lookahead_status = False
        self.it += 1


# replaces train_gpt.py's print0
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)


def _lr(step, initial_lr, power_c, power=1.0):
    t_end = SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)


def _muon_mu_at_step(step, train_steps):
    cd_start = train_steps - _MU_COOLDOWN_STEPS
    if step < _MU_WARMUP_STEPS:
        frac = step / max(_MU_WARMUP_STEPS, 1)
        return _MU_MIN + frac * (_MU_MAX - _MU_MIN)
    elif step > cd_start:
        frac = (step - cd_start) / max(_MU_COOLDOWN_STEPS, 1)
        return _MU_MAX - frac * (_MU_MAX - _MU_MIN)
    else:
        return _MU_MAX


def set_hparams(step):
    assert 0 <= step / SCHEDULE_STEPS < 1
    mu = _muon_mu_at_step(step, TRAIN_STEPS)
    for opt in hparam_optimizers:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], LR_POWER)
    for group in optimizer2.param_groups:
        group["mu"] = mu


def maybe_capture_ref_extrap_state(step: int):
    global ref_extrap_state, ref_extrap_captured_step
    if step != REF_CAPTURE_STEP or ref_extrap_state is not None:
        return
    ref_extrap_state = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if torch.is_floating_point(p)
    }
    ref_extrap_captured_step = step
    print0(f"captured_ref_extrap_state step:{step}", console=True)


def ref_extrap_gamma_for_step(step: int) -> float:
    return REF_GAMMA if step >= REF_APPLY_START_STEP else 0.0


def apply_ref_extrap_weights(gamma: float):
    if gamma == 0.0:
        return []
    if ref_extrap_state is None:
        raise RuntimeError("reference interpolation requested before capture")
    applied = []
    for name, p in model.named_parameters():
        ref = ref_extrap_state.get(name)
        if ref is None or not torch.is_floating_point(p):
            continue
        ref = ref.to(device=p.device, dtype=p.dtype)
        delta = (p.detach() - ref) * gamma
        p.add_(delta)
        applied.append((p, delta))
    return applied


def restore_ref_extrap_weights(applied):
    for p, delta in reversed(applied):
        p.sub_(delta)


def compute_validation_loss():
    val_loss = 0
    assert len(val_inputs) % mbs == 0
    for i in range(len(val_inputs) // mbs):
        val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    return val_loss / val_tokens


def ema_nesterov_suffix() -> str:
    return f" ema_beta:{optimizers[0].current_lookahead_stepsize:.5g}"


def ref_extrap_suffix(step: int) -> str:
    gamma = ref_extrap_gamma_for_step(step)
    if gamma == 0.0:
        return ""
    return f" ref_gamma:{gamma:g} ref_step:{ref_extrap_captured_step}"
