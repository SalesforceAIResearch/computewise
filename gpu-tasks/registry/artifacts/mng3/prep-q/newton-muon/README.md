# newton-muon

This directory contains a first valid Track 3 benchmark result for
[Newton-Muon](https://arxiv.org/abs/2604.01472). This is a per-optimizer result
rather than a global SOTA claim.

The submitted runs were launched with `train_steps = 3300` and logged validation
every 25 steps near the end of training. The earliest shared checkpoint that
passes the Track 3 significance rule is step 3275.

_(From the submission's README (20260505_newton_muon/README.md).)_

## How it differs from train_gpt.py

- Replaces: `CausalSelfAttention`, `MLP`, `Muon`.
- Adds: `accum_xtx_`, `attach_precond_stats`, `precond_stat_hook`, `enable_precond_hooks`.
- All of them are in `optimizer.py` in this folder, with the constants its run used (`hparams.md` lists them).
- Its reference run takes no command-line arguments.

The reference run's training setup and loop, from the model's construction to the end:

```python
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
attach_precond_stats(model)
model.compile(dynamic=False)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3300
    stop_after_step = 250  # short run: stop after this step's validation; the learning-rate schedule is unchanged

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    named_block_params = [(n, p) for n, p in model.named_parameters()
                          if n.startswith("blocks.") and p.ndim >= 2]
    qkv_params = [p for n, p in named_block_params
                  if n.endswith(".attn.q.weight") or n.endswith(".attn.k.weight") or n.endswith(".attn.v.weight")]
    attn_out_params = [p for n, p in named_block_params if n.endswith(".attn.proj.weight")]
    mlp_fc_params = [p for n, p in named_block_params if n.endswith(".mlp.fc.weight")]
    mlp_proj_params = [p for n, p in named_block_params if n.endswith(".mlp.proj.weight")]
    optimizer2 = Muon([dict(params=qkv_params, lr=0.020, weight_decay=0.025),
                       dict(params=attn_out_params, lr=0.025, weight_decay=0.025),
                       dict(params=mlp_fc_params, lr=0.030, weight_decay=0.0375),
                       dict(params=mlp_proj_params, lr=0.030, weight_decay=0.025)],
                      refresh_interval=64)
    optimizer2.attach_preconditioner()
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps or step == stop_after_step:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        optimizer2.global_step = step
        precond_hooks = enable_precond_hooks(model) if optimizer2.precond_flag_for_step(step) else []
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for h in precond_hooks:
            h.remove()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
```

## Our run of it

A short screening run: the first 250 optimizer steps of this schedule, stopping after the validation at step 250. Keep the schedule's own length (3300 steps and its decay constants) so the first 250 steps are the same as a full run's; only the stop is earlier. One GPU.
