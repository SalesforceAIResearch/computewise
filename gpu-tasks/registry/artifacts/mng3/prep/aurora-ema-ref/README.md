# aurora-ema-ref

This result uses the Aurora/EMA-Nesterov base from PR #309 with a fixed
reference-interpolated output rule. It captures reference weights at step 2375,
uses `gamma=-0.075` from step 2850 onward, and reports the fixed stopping point
K=2860.

The archived script hardcodes the submitted hyperparameter defaults. It reads
`--seed` to select the random seed for a reproducibility run; other command-line
arguments only control logging.

_(From the submission's README (20260525_aurora_ema_ref/README.md).)_

## How it differs from train_gpt.py

- Replaces: `distributed_data_generator`, `RMSNorm`, `CausalSelfAttention`, `zeropower_via_newtonschulz5`, `muon_update`, `Muon`, `print0`.
- Adds: `norm`, `gram_frobenius_norm_estimate`, `_ns_inner`, `scale_to_unit_operator_norm`, `should_soap_param`, `is_v_param`, `is_attn_param`, `scale_radial_update`, `target_radius_after_update`, `rescale_to_radius`, `soap_eigenbasis`, `soap_basis_qr`, `soap_precondition_momentum`, `soap_update_preconditioner`, `norm_preserving_soap_blend`, `_linear_ramp`, `contra_coeff_for_step`, `EMA_Nesterov`, `_lr`, `_muon_mu_at_step`, `set_hparams`, `maybe_capture_ref_extrap_state`, `ref_extrap_gamma_for_step`, `apply_ref_extrap_weights`, `restore_ref_extrap_weights`, `compute_validation_loss`, `ema_nesterov_suffix`, `ref_extrap_suffix`.
- All of them are in `optimizer.py` in this folder, with the constants its run used (`hparams.md` lists them).
- Its reference run takes these arguments: `parser.add_argument("--seed", type=int, default=0)`; `parser.add_argument("--log-dir", type=Path, default=Path("logs"))`; `parser.add_argument("--log-file", type=Path, default=None)`; `parser.add_argument("--progress-interval", type=int, default=100)`.

The reference run's training setup and loop, from the model's construction to the end:

```python
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = TRAIN_STEPS
run_stop_step = RUN_STOP_STEP
stop_after_step = 250  # short run: stop after this step's validation; the learning-rate schedule is unchanged
val_regular_interval = 125
extra_val_steps = {2850, 2860}
print0(f"Using run_stop_step={run_stop_step} train_steps={train_steps}")

# Depth-scaled MLP input projection init from the PR309/Aurora base.
_NUM_BLOCKS = len(model.blocks)

# Initialize active method components: zero projection matrices, zero all
# non-projection biases, and apply the PR309 depth-scaled MLP-fc init.
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()
    elif name.endswith("bias") and ZERO_NONPROJ_BIASES:
        p.data.zero_()

for l_idx, block in enumerate(model.blocks):
    ramp = l_idx / (_NUM_BLOCKS - 1) if _NUM_BLOCKS > 1 else 0.0
    block.mlp.fc.weight.data.mul_(1.0 - DI_FC_ALPHA * ramp)

# CGI alpha is intentionally fixed to zero for this result; norm gains remain 1.

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.99), eps=1e-10, weight_decay=0, fused=True)
# Skylight-001: NorMuon-lite (per-row variance) + u/w-floor=0.35 + lr=0.0375 + wd=0.025.
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2],
                  lr=MUON_LR, mu=MU)
hparam_optimizers = [optimizer1, optimizer2]
optimizers = [EMA_Nesterov(
    [p for p in model.parameters()],
    hparam_optimizers,
    lookahead_stepsize=EMA_NESTEROV_STEPSIZE,
    use_scheduled_lookahead_stepsize=True,
    lookahead_ema=EMA_NESTEROV_LOOKAHEAD_EMA,
    prefill_steps=EMA_NESTEROV_PREFILL_STEPS,
    rest_steps=EMA_NESTEROV_REST_STEP,
)]
assert set(p for opt in hparam_optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in hparam_optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
optimizer1.param_groups[0]["power_c"] = ADAM_EMBED_POWER_C
optimizer1.param_groups[1]["power_c"] = ADAM_PROJ_POWER_C
optimizer1.param_groups[2]["power_c"] = ADAM_OTHER_POWER_C
optimizer2.param_groups[0]["power_c"] = MUON_POWER_C

# learning rate schedule: stable then decay
def _lr(step, initial_lr, power_c, power=1.0):
    t_end = SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)

# v49: v15 Muon mu schedule (warmup 0.85->0.95 over 300 steps, cooldown 0.95->0.85 over last 50)
_MU_MIN = 0.85
_MU_MAX = 0.95
_MU_WARMUP_STEPS = 300
_MU_COOLDOWN_STEPS = MUON_COOLDOWN_STEPS

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




ref_extrap_state = None
ref_extrap_captured_step = None

@torch.no_grad()
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

@torch.no_grad()
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

@torch.no_grad()
def restore_ref_extrap_weights(applied):
    for p, delta in reversed(applied):
        p.sub_(delta)

@torch.no_grad()
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


########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(run_stop_step + 1):

    # --------------- VALIDATION SECTION -----------------
    should_validate = (
        step == train_steps
        or step == run_stop_step
        or (step > 0 and step % val_regular_interval == 0)
        or step in extra_val_steps
    )
    if should_validate:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        ref_extrap_applied = apply_ref_extrap_weights(ref_extrap_gamma_for_step(step))
        model.eval()
        val_loss = compute_validation_loss()
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms"
               + ema_nesterov_suffix()
               + ref_extrap_suffix(step), console=True)
        restore_ref_extrap_weights(ref_extrap_applied)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == run_stop_step or step == stop_after_step:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for opt in optimizers:
        if hasattr(opt, "nesterov_step"):
            opt.nesterov_step()
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        # NaN guard: catch divergence the step it happens, not 750 steps later.
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite train loss at step {step} mb {i}: {loss.item()}")
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    maybe_capture_ref_extrap_state(step + 1)
    model.zero_grad(set_to_none=True)
    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms",
               console=True, log=False)

dist.destroy_process_group()
```

## Our run of it

A short screening run: the first 250 optimizer steps of this schedule, stopping after the validation at step 250. Keep the schedule's own length (2860 steps and its decay constants) so the first 250 steps are the same as a full run's; only the stop is earlier. One GPU.
