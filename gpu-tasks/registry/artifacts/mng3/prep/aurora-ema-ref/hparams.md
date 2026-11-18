# aurora-ema-ref: hyperparameters

From the reference run's script (20260525_aurora_ema_ref).

| setting | value |
|---|---|
| training steps (`train_steps`) | `TRAIN_STEPS` |
| our screening run stops after step | `250` |
| `SEED` | `args.seed` |
| `LOG_DIR` | `args.log_dir` |
| `LOG_FILE` | `args.log_file` |
| `TRAIN_PROGRESS_INTERVAL` | `args.progress_interval` |
| `TRAIN_STEPS` | `2900` |
| `RUN_STOP_STEP` | `2860` |
| `SCHEDULE_STEPS` | `2980` |
| `LR_POWER` | `1.2` |
| `REF_CAPTURE_STEP` | `2375` |
| `REF_APPLY_START_STEP` | `2850` |
| `REF_GAMMA` | `-0.075` |
| `ADAM_EMBED_POWER_C` | `4.976805410800738e-05` |
| `ADAM_PROJ_POWER_C` | `5.184172302917436e-07` |
| `ADAM_OTHER_POWER_C` | `1.6589351369335795e-06` |
| `MUON_POWER_C` | `3.3169534699576625e-06` |
| `CONTRA_MUON_COEFF` | `-0.2` |
| `CONTRA_HOLD_END_STEP` | `0` |
| `CONTRA_TO_NORMAL_END_STEP` | `2500` |
| `MU` | `0.95` |
| `MUON_LR` | `0.0375` |
| `MUON_COOLDOWN_STEPS` | `100` |
| `TARGET_UW` | `0.3825` |
| `NOR_BETA2` | `1.0` |
| `SOAP_BETA2` | `0.90` |
| `SOAP_PRECONDITION_FREQUENCY` | `10` |
| `SOAP_DENOM_POWER` | `0.50` |
| `SOAP_BLEND` | `1.00` |
| `V_SOAP_BLEND` | `0.95` |
| `ATTN_SOAP_DENOM_FLOOR` | `0.55` |
| `RADIAL_OUTWARD_SCALE` | `0.5` |
| `RADIAL_INWARD_SCALE` | `1.0` |
| `CGI_ALPHA` | `0.0` |
| `ZERO_NONPROJ_BIASES` | `True` |
| `DI_FC_ALPHA` | `0.30` |
| `EMA_NESTEROV_STEPSIZE` | `0.3` |
| `EMA_NESTEROV_LOOKAHEAD_EMA` | `0.99` |
| `EMA_NESTEROV_PREFILL_STEPS` | `300` |
| `EMA_NESTEROV_REST_STEP` | `TRAIN_STEPS - 950` |
| `_AURORA_K` | `3` |
| `_AURORA_BETA` | `0.25` |
| `_AURORA_EPS` | `1e-7` |
| `_NUM_BLOCKS` | `len(model.blocks)` |
| `_MU_MIN` | `0.85` |
| `_MU_MAX` | `0.95` |
| `_MU_WARMUP_STEPS` | `300` |
| `_MU_COOLDOWN_STEPS` | `MUON_COOLDOWN_STEPS` |
