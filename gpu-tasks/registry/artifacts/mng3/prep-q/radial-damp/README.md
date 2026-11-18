# radial-damp

This submission builds on PR291's Contra-Muon to Soft-Muon setup and adds
post-step radial damping. After applying the optimizer update, the hidden matrix
parameter norm is rescaled so outward radial movement is dampened while inward
movement is preserved according to the constants logged in each run.

The submitted step count is **2990**. The logs include a fixed validation grid
at 2990, 2995, 2999, 3000, 3010, and 3020 for all seeds.

Across 11 non-cherry-picked seed logfiles, the step 2990 mean validation loss is
3.27866818. Under the Track 3 README criterion,
`(3.28 - mu) * sqrt(n) = 0.00441714`, which exceeds the required `0.004`
threshold. Equivalently, using sigma=0.0013 gives `z=3.3978` and one-sided
`p=0.00034`, satisfying the p<0.001 criterion at 2990 steps.

_(From the submission's README (20260511_dampen_radial_gradient_component/README.md).)_

## How it differs from train_gpt.py

- Replaces: `distributed_data_generator`, `RMSNorm`, `CausalSelfAttention`, `zeropower_via_newtonschulz5`, `muon_update`, `Muon`, `print0`.
- Adds: `norm`, `gram_frobenius_norm_estimate`, `_soft_coefficients`, `zeropower_frobenius_norm_like`, `soft_via_newtonschulz5`, `scale_to_unit_operator_norm`, `should_soap_param`, `is_attn_proj_param`, `is_attn_param`, `is_v_param`, `param_matches_spec`, `tensor_cosine`, `trust_gate`, `early_trust_floor_for_step`, `bounded_trust_gate`, `attention_soap_blend_for_step`, `norm_preserving_blend`, `scale_radial_update`, `target_radius_after_update`, `rescale_to_radius`, `soap_eigenbasis`, `soap_basis_qr`, `soap_precondition_momentum`, `soap_update_preconditioner`, `_linear_ramp`, `contra_coeff_for_step`, `soft_blend_for_step`, `_lr`, `set_hparams`.
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
stop_after_step = 250  # short run: stop after this step's validation; the learning-rate schedule is unchanged
val_regular_interval = 125
extra_val_steps = {2990, 2995, 2999, 3000, 3010, 3020}

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
# Skylight-001: NorMuon-lite (per-row variance) + u/w-floor=0.35 + lr=0.0375 + wd=0.025.
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2],
                  lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY, mu=MU)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
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

def set_hparams(step):
    progress = step / FINAL_SCHEDULE_STEPS
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = _lr(step, group["initial_lr"], group["power_c"], FINAL_LR_POWER)




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

A short screening run: the first 250 optimizer steps of this schedule, stopping after the validation at step 250. Keep the schedule's own length (3020 steps and its decay constants) so the first 250 steps are the same as a full run's; only the stop is earlier. One GPU.
