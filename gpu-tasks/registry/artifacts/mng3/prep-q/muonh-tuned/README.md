# muonh-tuned

train_gpt_simple_muonh.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.

Differs from `train_gpt_simple.py` only in the init and the optimizer: the matrix-parameter
Muon update is replaced by MuonH (the same Newton-Schulz orthogonalised direction, applied
via a hyperball projection that preserves the Frobenius norm of every hidden 2D weight
matrix at every step), and the residual-side projections / mlp.fc are initialised with
per-module multipliers on the default Kaiming init so MuonH has non-zero matrices to operate
on from step 0.

_(From the reference script's docstring (the submission has no README).)_

## How it differs from train_gpt.py

- Adds: `scale_invariant_update_`, `MuonH`.
- All of them are in `optimizer.py` in this folder, with the constants its run used (`hparams.md` lists them).
- Its reference run takes no command-line arguments.

The reference run's training setup and loop, from the model's construction to the end:

```python
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3250
    stop_after_step = 250  # short run: stop after this step's validation; the learning-rate schedule is unchanged

    # initialize model parameters. Per-module multipliers on the default nn.Linear Kaiming-uniform
    # init (std = 1/sqrt(3*fan_in), so ~0.0208 for fan_in=768 and ~0.0104 for fan_in=3072):
    #   - attn.proj.weight (fan_in=768):  default × 1.25 → std ≈ 0.026
    #   - mlp.proj.weight  (fan_in=3072): default × 3.0  → std ≈ 0.031
    #   - mlp.fc.weight    (fan_in=768):  default × 1.5  → std ≈ 0.031
    # qkv weights keep their default init. The vocab head (proj.weight) and all "proj" biases are
    # zeroed so initial logits are 0.
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
        if name.endswith(".attn.proj.weight"):
            w.mul_(1.25)
        elif name.endswith(".mlp.proj.weight"):
            w.mul_(3.0)
        elif name.endswith(".mlp.fc.weight"):
            w.mul_(1.5)

    # create the optimizer(s)
    # aux AdamW hyperparameters grafted from the tuned baseline in
    # https://github.com/KellerJordan/modded-nanogpt/pull/323 (embed 0.3->0.7,
    # head 1/320->0.004, scalars 0.01->0.015, weight_decay 0->0.001)
    # per-module aux lr tuning around x1.3 center: head_hi (others at x1.3)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.910),
                        dict(params=[model.proj.weight], lr=0.0064),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.0195)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = MuonH([p for p in model.blocks.parameters() if p.ndim == 2], lr=0.018)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    for group in optimizer1.param_groups:
        group["cooldown_frac"] = 0.85  # aux cooldown sweep (PR #323 baseline uses 0.7)
    for group in optimizer2.param_groups:
        group["cooldown_frac"] = 1.0

    # learning rate schedule: stable then decay. The h (MuonH) groups use full linear cooldown
    # over the entire run (cooldown_frac=1.0); the aux (AdamW) group uses the tuned baseline's
    # cooldown_frac=0.7, matching the schedule under which the PR #323 aux lrs were tuned.
    def set_hparams(step):
        progress = step / train_steps
        assert 0 <= progress < 1
        for opt in optimizers:
            for group in opt.param_groups:
                if progress < 1 - group["cooldown_frac"]:
                    eta = 1.0
                else:
                    eta = (1 - progress) / group["cooldown_frac"]
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
        if step == train_steps or step % 125 == 0:
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
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
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

A short screening run: the first 250 optimizer steps of this schedule, stopping after the validation at step 250. Keep the schedule's own length (3250 steps and its decay constants) so the first 250 steps are the same as a full run's; only the stop is earlier. One GPU.
