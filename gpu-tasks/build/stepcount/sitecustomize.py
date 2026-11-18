# Step counter for registry jobs. Put this directory first on PYTHONPATH and set STEPCOUNT=<prefix>.
# Every Python process then writes <prefix>.<pid>.json at exit with the number of optimizer.step() calls its Hugging Face
# Trainer made, which is what gpu-mock's replay paces. Unless STEPCOUNT_COUNT_ONLY=1, no model is computed: inside train() the
# instance's training_step returns a zero loss (no forward, no backward) and evaluate, checkpoint saves and save_model are
# no-ops, so the dataloader, sharding across ranks, gradient accumulation and step counting run exactly as in the job.
# Instance attributes are used so a Trainer subclass that overrides those methods is covered too.
# Heavy (tokenises and iterates the whole dataset): run on the ComputeWise VM, never on the controller.
import os
import sys

if os.environ.get("STEPCOUNT"):
    import atexit
    import json

    _n = {"argv0": os.path.basename(sys.argv[0]) if sys.argv else "", "rank": os.environ.get("RANK"),
          "local_rank": os.environ.get("LOCAL_RANK"), "world_size": os.environ.get("WORLD_SIZE"),
          "count_only": os.environ.get("STEPCOUNT_COUNT_ONLY") == "1", "optimizer_steps": 0, "training_step_calls": 0,
          "evaluate_calls": 0, "checkpoint_saves": 0, "save_model_calls": 0, "train_entered": False,
          "global_step": None, "max_steps": None, "patch_error": None}

    def _dump():
        with open(f"{os.environ['STEPCOUNT']}.{os.getpid()}.json", "w") as fh:
            json.dump(_n, fh)

    atexit.register(_dump)
    if _n["argv0"].endswith(".py"):
        try:
            import types

            import torch
            from torch.optim.optimizer import register_optimizer_step_post_hook

            def _hook(opt, args, kwargs):
                _n["optimizer_steps"] += 1

            register_optimizer_step_post_hook(_hook)
            from transformers import Trainer

            def _training_step(self, model, inputs, *a, **k):
                _n["training_step_calls"] += 1
                return torch.tensor(0.0, device=self.args.device)

            def _evaluate(self, *a, **k):
                _n["evaluate_calls"] += 1
                return {}

            def _save_checkpoint(self, *a, **k):
                _n["checkpoint_saves"] += 1

            def _save_model(self, *a, **k):
                _n["save_model_calls"] += 1

            _train = Trainer.train

            def _train_wrap(self, *a, **k):
                _n["train_entered"] = True
                if not _n["count_only"]:
                    for name, fn in (("training_step", _training_step), ("evaluate", _evaluate),
                                     ("_save_checkpoint", _save_checkpoint), ("save_model", _save_model)):
                        setattr(self, name, types.MethodType(fn, self))
                try:
                    return _train(self, *a, **k)
                finally:
                    _n["global_step"], _n["max_steps"] = self.state.global_step, self.state.max_steps

            Trainer.train = _train_wrap
        except Exception as e:  # report, never block the interpreter
            _n["patch_error"] = repr(e)
