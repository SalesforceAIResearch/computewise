# ema-nesterov-aurora

radial_scale_then_radius_correct.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.

---

dampen radial gradient component

---

This submission incorporates features from the following previous submissions

@nilin PR291

@kumarkrishna PR274 / Skylight-001
  NorMuon-lite row/column variance normalization
  u/w floor postprocessing
  lr=0.0375 style Muon setup

@nilin (me): PR275 / Contra-Muon
  Introduces Contra-Muon update term

@samacqua: PR278 / MLP SOAP preconditioning 
  SOAP preconditioning machinery / MLP SOAP idea.
  Our script uses that SOAP machinery and extends the selected SOAP set to MLP+V.

@yash-oai: PR287 / power law LR schedule
  Power-law LR schedule PowerCool:
  min(flat_lr, c * (t_end - step)^1.2)
  PR287-style cooldown constants / schedule tuning.

The SOAP-like preconditioning from samacqua / PR 278 is also applied to the attention value-projection (V) matrices in this submission.

_(From the reference script's docstring (the submission has no README).)_

## How it differs from train_gpt.py

- Replaces: `distributed_data_generator`, `RMSNorm`, `CausalSelfAttention`, `zeropower_via_newtonschulz5`, `muon_update`, `Muon`, `print0`.
- Adds: `norm`, `gram_frobenius_norm_estimate`, `_ns_inner`, `_soft_coefficients`, `zeropower_frobenius_norm_like`, `soft_via_newtonschulz5`, `scale_to_unit_operator_norm`, `should_soap_param`, `is_attn_proj_param`, `is_attn_param`, `is_v_param`, `param_matches_spec`, `tensor_cosine`, `trust_gate`, `early_trust_floor_for_step`, `bounded_trust_gate`, `attention_soap_blend_for_step`, `norm_preserving_blend`, `scale_radial_update`, `target_radius_after_update`, `rescale_to_radius`, `soap_eigenbasis`, `soap_basis_qr`, `soap_precondition_momentum`, `soap_update_preconditioner`, `_linear_ramp`, `contra_coeff_for_step`, `soft_blend_for_step`, `EMA_Nesterov`, `_lr`, `_muon_mu_at_step`, `set_hparams`.
- All of them are in `optimizer.py` in this folder, with the constants its run used (`hparams.md` lists them).
- Its reference run takes these arguments: `parser.add_argument("--seed", type=int, default=0)`.

The reference run's training setup and loop, from the model's construction to the end:

```python
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = FINAL_TRAIN_STEPS
stop_after_step = 125  # short run: stop after this step's validation; the learning-rate schedule is unchanged
val_regular_interval = 125
extra_val_steps = {2820, 2830, 2840, 2850, 2860, 2870, 2880, 2890, 2895, 2900, 2910, 2920, 2930, 2940, 2950, 2960, 2965, 2970, 2975, 2980, 2985, 2990, 2995, 2999, 3000, 3010, 3020}

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# v44: v13 depth-scaled mlp.fc init alpha=0.30 (ported from opus v15)
_DI_FC_ALPHA = 0.30
_NUM_BLOCKS = len(model.blocks)
with torch.no_grad():
    for l_idx, block in enumerate(model.blocks):
        ramp = l_idx / (_NUM_BLOCKS - 1) if _NUM_BLOCKS > 1 else 0.0
        s_l = 1.0 - _DI_FC_ALPHA * ramp
        block.mlp.fc.weight.data.mul_(s_l)

# v44: v14 CGI Rademacher channel-gain split alpha=0.14 (ported from opus v15)
_CGI_ALPHA = 0.14
with torch.no_grad():
    for block in model.blocks:
        s = (torch.randint(0, 2, block.norm1.gains.shape,
                           device=block.norm1.gains.device, dtype=torch.float32) * 2 - 1)
        block.norm1.gains.data.copy_((1.0 - _CGI_ALPHA * s).to(block.norm1.gains.dtype))
        block.norm2.gains.data.copy_((1.0 + _CGI_ALPHA * s).to(block.norm2.gains.dtype))

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.99), eps=1e-10, weight_decay=0, fused=True)
# Skylight-001: NorMuon-lite (per-row variance) + u/w-floor=0.35 + lr=0.0375 + wd=0.025.
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2],
                  lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY, mu=MU)
optimizers = [optimizer1, optimizer2]
optimizers = [EMA_Nesterov(
        [p for p in model.parameters()],
        optimizers,
        lookahead_stepsize=0.3,
        use_scheduled_lookahead_stepsize=True,
        lookahead_ema=0.99,
        prefill_steps=300,
        rest_steps=train_steps - 950,
    )]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers[0].inner_optimizer:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
optimizer1.param_groups[0]["power_c"] = ADAM_EMBED_POWER_C
optimizer1.param_groups[1]["power_c"] = ADAM_PROJ_POWER_C
optimizer1.param_groups[2]["power_c"] = ADAM_OTHER_POWER_C
optimizer2.param_groups[0]["power_c"] = MUON_POWER_C

# learning rate schedule: stable then decay
def _lr(step, initial_lr, power_c, power=1.0):
    t_end = FINAL_SCHEDULE_STEPS
    flat_lr = initial_lr
    downward_lr = power_c * max(0.0, t_end - step) ** power
    return min(flat_lr, downward_lr)

# v49: v15 Muon mu schedule (warmup 0.85->0.95 over 300 steps, cooldown 0.95->0.85 over last 50)
_MU_MIN = 0.85
_MU_MAX = 0.95
_MU_WARMUP_STEPS = 300
_MU_COOLDOWN_STEPS = 100

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
    progress = step / FINAL_SCHEDULE_STEPS
    assert 0 <= progress < 1
    mu = _muon_mu_at_step(step, FINAL_TRAIN_STEPS)
    for opt in optimizers[0].inner_optimizer:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)
    for group in optimizer2.param_groups:
        group["mu"] = mu




########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    should_validate = (
        step == train_steps
        or (step > 0 and step % val_regular_interval == 0)
        or step in extra_val_steps
    )
    if should_validate:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps or step == stop_after_step:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    optimizers[0].nesterov_step()
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
    model.zero_grad(set_to_none=True)
    if TRAIN_PROGRESS_INTERVAL > 0 and (step + 1) % TRAIN_PROGRESS_INTERVAL == 0:
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
```

## Our run of it

A short screening run: the first 125 optimizer steps of this schedule, stopping after the validation at step 125. Keep the schedule's own length (2900 steps and its decay constants) so the first 125 steps are the same as a full run's; only the stop is earlier. One GPU.
