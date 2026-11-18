# soda-fade: hyperparameters

From the reference run's script (20260518_soda_fade_3000_n9).

| setting | value |
|---|---|
| training steps (`train_steps`) | `FINAL_TRAIN_STEPS` |
| our screening run stops after step | `1000` |
| `SODA_HIDDEN` | `True` |
| `SODA_LAMBDA_SCALE` | `1.0` |
| `SODA_FADE_START_STEP` | `2000` |
| `SODA_FADE_END_STEP` | `2750` |
| `SODA_FADE_SHAPE` | `"cosine"` |
| `SEED` | `args.seed` |
| `FINAL_TRAIN_STEPS` | `3040` |
| `FINAL_SCHEDULE_STEPS` | `3105` |
| `FINAL_LR_POWER` | `1.2` |
| `ADAM_EMBED_POWER_C` | `4.976805410800738e-05` |
| `ADAM_PROJ_POWER_C` | `5.184172302917436e-07` |
| `ADAM_OTHER_POWER_C` | `1.6589351369335795e-06` |
| `MUON_POWER_C` | `3.3169534699576625e-06` |
| `FINAL_MUON_WD` | `0.025` |
| `EXPERIMENT_NAME` | `"contra-muon-to-soft-muon"` |
| `EXPERIMENT_INTUITION` | `"Interpolate Muon update from Contra Muon to normal Muon, then from normal Muon to p=0.1 soft Muon."` |
| `CONTRA_MUON_COEFF` | `-0.2` |
| `SOFT_MUON_P` | `0.1` |
| `SOFT_MUON_SCALE` | `"none"` |
| `SOFT_MUON_INPUT_NORM` | `"frobenius_schatten4"` |
| `SOFT_MUON_CEIL` | `0.80` |
| `CONTRA_HOLD_END_STEP` | `0` |
| `CONTRA_TO_NORMAL_END_STEP` | `2000` |
| `NORMAL_TO_SOFT_START_STEP` | `2500` |
| `NORMAL_TO_SOFT_END_STEP` | `3010` |
| `MU` | `0.95` |
| `MUON_LR` | `0.0375` |
| `MUON_WEIGHT_DECAY` | `FINAL_MUON_WD` |
| `TARGET_UW` | `0.3825` |
| `SOAP_TARGET_UW` | `TARGET_UW` |
| `NONSOAP_TARGET_UW` | `TARGET_UW` |
| `NOR_BETA2` | `0.95` |
| `SOAP_BETA2` | `0.90` |
| `SOAP_PRECONDITION_FREQUENCY` | `10` |
| `SOAP_DENOM_POWER` | `0.50` |
| `SOAP_BLEND` | `1.00` |
| `SOAP_UPDATE_BEFORE_USE` | `False` |
| `SOAP_PARAM_MODE` | `"mlp_plus_v"` |
| `ATTN_SOAP_DENOM_FLOOR` | `0.55` |
| `ATTN_SOAP_BLEND` | `1.00` |
| `V_SOAP_BLEND` | `0.95` |
| `V_SOAP_BLEND_RAMP_END_STEP` | `0` |
| `ATTN_EARLY_TRUST_FLOOR` | `0.45` |
| `ATTN_EARLY_TRUST_CAP` | `0.85` |
| `ATTN_TRUST_FLOOR_END_STEP` | `1375` |
| `ATTN_TRUST_FLOOR_FADE_END_STEP` | `1625` |
| `ATTN_TRUST_MIN_AGREE` | `0.20` |
| `ATTN_TRUST_MIN_GRAD_ALIGN` | `0.00` |
| `ATTN_TRUST_POWER` | `1.00` |
| `ATTN_SOAP_FADE_START_STEP` | `1000000000` |
| `ATTN_SOAP_FADE_END_STEP` | `1000000000` |
| `NO_CONTRA_PARAM` | `""` |
| `NO_SOFTMUON_PARAM` | `""` |
| `TRAIN_PROGRESS_INTERVAL` | `0` |
| `LOG_DIR` | `Path("experiment-logs/all-runs/2026-05-09/validation-ceil-0.8")` |
