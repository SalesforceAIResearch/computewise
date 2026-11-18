"""Tensor.type() string-query compatibility (next/ r8.4, from Astra's NeMo
diagnosis 2026-11-02, step-probe-3). A no-dtype `.type()` QUERY on a cuda-labelled
tensor must describe the emulated device — `torch.cuda.<DType>Tensor`, not the CPU
`torch.<DType>Tensor` — exactly as `.device`/`.is_cuda` already do. Megatron's
distributed optimizer string-compares `.type()` to pick its parameter branch
(distrib_optimizer.py), so the CPU string made it reject every cuda param.

Astra's proposal wrapped `FakeCudaTensor.type`; that is correct on the NeMo image
(torch 2.5, where a fake tensor's instance class IS FakeCudaTensor) but NOT on the
2.11 boxes, where a cuda tensor is a plain torch.Tensor (FakeCudaTensor is only a
virtual/ABC subclass — isinstance True but type(t) is torch.Tensor), so a method
set on FakeCudaTensor is never found by MRO on t.type. Wrapping `torch.Tensor.type`
and keying on the live `self.is_cuda` works on every torch the patch supports: a
cuda-labelled tensor gets the torch.cuda.* string, a CPU tensor is untouched, and a
conversion call (args → returns a tensor) delegates unchanged. Installed from
cuda_hooks.apply() after _fix_tensor_class.
"""


def install():
    try:
        import torch
    except Exception:
        return
    original = torch.Tensor.type
    if getattr(original, "_nvrt_cuda_type_query", False):
        return

    def tensor_type(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # isinstance(result, str) is the discriminator: a STRING QUERY (no dtype,
        # dtype=None, or an explicit None) returns the type string; a CONVERSION
        # call returns a tensor and passes through untouched. Do NOT gate on
        # (not args and not kwargs) — type(None) / type(dtype=None) are valid
        # string queries that must also report the cuda name (Astra, 2026-11-02).
        if (isinstance(result, str) and getattr(self, "is_cuda", False)
                and result.startswith("torch.") and not result.startswith("torch.cuda.")):
            return "torch.cuda." + result[len("torch."):]
        return result

    tensor_type._nvrt_cuda_type_query = True
    torch.Tensor.type = tensor_type
