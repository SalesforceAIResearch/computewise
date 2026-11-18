"""Test-only: run an UNMODIFIED training script and capture its model's state just
before its training loop starts, then stop. For the SF checkpoint-collision W0's
artifact-identity question: does each W0 checkpoint's
model_digest equal the digest of its own seeded initial weights?

  python3 capture_init_state.py <out.pt> <script> [script args...]

The script runs as __main__ through runpy.run_path, with sys.argv set to
[<script>, args...], so gpu-mock's patch sees the same entry script, entry hash and
argv as the W0 launch did, and replay engages (or not) exactly as it would. A trace
function limited to the script's own module frame waits for the first top-level
`for step in ...:` line (found from the script's AST, not hard-coded). There it calls
torch.save({"model": model.state_dict()}, out) and raises SystemExit, before the loop
body (the first forward, backward or optimizer step) runs. The script file is not
edited; nothing else is traced. Exit code 0 means captured; 3 means the loop line was
never reached."""
import ast
import os
import runpy
import sys


def loop_line(path: str) -> int | None:
    tree = ast.parse(open(path, "rb").read(), path)
    for node in tree.body:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step":
            return node.lineno
    return None


def main() -> int:
    out, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    real = os.path.realpath(script)
    line = loop_line(real)
    if line is None:
        print(f"CAPTURE no top-level `for step` loop in {script}", flush=True)
        return 3
    done = {"v": False}

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line and not done["v"]:
            done["v"] = True
            import torch
            model = frame.f_globals["model"]
            torch.save({"model": model.state_dict()}, out)
            print(f"CAPTURE saved model state at line {line} of {script} -> {out}", flush=True)
            sys.settrace(None)
            raise SystemExit(0)
        return local

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "<module>" \
                and os.path.realpath(frame.f_code.co_filename) == real:
            return local
        return None

    sys.argv = [script] + list(args)
    sys.settrace(tracer)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.settrace(None)
    if not done["v"]:
        print(f"CAPTURE loop line {line} never reached", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
