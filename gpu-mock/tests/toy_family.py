"""The placeholder surrogate family both surrogate tests use (tests only; the
real families are gpu-tasks' fits under registry/surrogates/)."""

SPEC = {"family": "toy-gpt", "version": 1, "scripts": ["design/train_gpt.py"],
        "knobs": {"lr": {"flag": "--lr", "default": 0.02, "range": [0.001, 0.2]},
                  "model_dim": {"flag": "--model_dim", "fingerprint": "model_dim", "default": 768},
                  "num_layers": {"flag": "--num_layers", "fingerprint": "num_layers", "default": 12},
                  "batch_tokens": {"flag": "--batch_tokens", "fingerprint": "global_batch_tokens", "default": 524288},
                  "train_steps": {"flag": "--steps", "default": 3250}},
        "init_loss": {"value": 10.96, "sd": 0.02},
        "loss": {"form": "power_law", "params": {"L_inf": 2.90, "A": 11.5, "t0": 40, "alpha": 0.45},
                 "effects": [{"knob": "model_dim", "kind": "log_ratio", "ref": 768, "coef": -0.32},
                             {"knob": "lr", "kind": "log_quadratic", "opt": 0.02, "curv": 0.10}],
                 "noise": {"kind": "ar1", "rho": 0.6, "sd_at_step": [[0, 0.02], [125, 0.004], [1000, 0.0025], [3000, 0.0012]]},
                 "divergence": {"when": {"lr_above": 0.12}, "after_steps": 30, "then": "nan"}},
        "train_loss": {"gap_below_val": 0.03, "noise_sd": 0.03},
        "timing": {"steady_step_s": {"form": "flops", "flops_per_token_per_param": 6, "achieved_tflops": 290,
                                     "overhead_s": 0.008, "ref": {"gpu": "H100", "world_size": 8}},
                   "startup_in_train_s": {"form": "const", "value": 184.0},
                   "eval_s_per_validation": {"form": "ratio_of_step", "factor": 6.67}},
        "memory": {"peak_bytes": {"form": "linear", "bytes_per_param": 18, "bytes_per_batch_token": 48, "base_bytes": 1.5e9}},
        "validation_steps": {"every": 125, "and_last": True}}
