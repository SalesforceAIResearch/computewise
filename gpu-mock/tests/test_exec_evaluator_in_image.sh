#!/usr/bin/env bash
# An evaluator exec'd into an agent's own globals dict (G-kvreph-haiku-L0-m3,
# 2026-10-27): `python3 - <<EOF ... exec(compile(src, path, "exec"), g)` with
# g = {"__name__": "__main__", "__file__": path}. The lookup matches exact, since
# __file__ names the real script, and replay engages; in 3ec0 the per-example
# seams were looked up in sys.modules["__main__"] (the stdin module), were not
# found, and the real decode ran against the replay forward stub and crashed.
# Three launches on one daemon, of a stand-in whose generate function calls the
# model's forward and reads .past_key_values, the way MLS-Bench's does:
#   S  the stand-in run as a script: served, the record's score printed (as 3ec0);
#   X  the same source exec'd into a dict with __file__ set: served from the
#      seams found in that dict, the record's score printed, no exception;
#   Y  exec'd into a dict WITHOUT the seams' names reachable (the score function
#      renamed): replay withdrawn, the job runs on the modelled path to its end,
#      no AttributeError, a replay_disengaged event naming the reason.
# Usage (on the VM):  sudo IMG=sha256:<box id> [PATCH_SRC=1] bash tests/test_exec_evaluator_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro" -v "$G/tests:/gm/tests:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work/m /tmp/out; cd /tmp/work/m
cat > ev.py <<PY
import argparse, time, torch
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.l = torch.nn.Linear(8, 8)
    def forward(self, x):
        class Out: pass
        o = Out(); o.logits = self.l(x); o.past_key_values = ("cache",)
        return o
def generate_with_selection(model, example):
    out = model(torch.randn(1, 8, device="cuda"))
    cache = out.past_key_values          # the line the replay stub broke
    return "pred", {"retained_fraction": 1.0}
def score_prediction(example, prediction):
    return 100.0 if example % 2 else 0.0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workload", default="gsm8k"); a = ap.parse_args()
    model = Tiny().cuda()
    t0 = time.perf_counter(); scores = []
    for ex in range(10):
        pred, tr = generate_with_selection(model, ex)
        scores.append(score_prediction(ex, pred))
    print(f"TEST_METRICS: final_score={sum(scores)/len(scores):.6f} runtime_seconds={time.perf_counter()-t0:.6f}", flush=True)
if __name__ == "__main__":
    main()
PY
EH=$(sha256sum ev.py | cut -c1-64)
python - "$EH" <<PY
import json, sys
json.dump({"identity": {"record_id": "standin-exec", "entry_hash": sys.argv[1], "argv": ["ev.py", "--workload", "gsm8k"], "match_policy": "exact_only", "replicate_group": "standin-exec"},
           "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": 5.0, "provenance": "recorded"}},
           "memory": {"peak_allocated_bytes": 1024 ** 3, "peak": 1024 ** 3}, "series": {},
           "result": {"final_score": {"value": 30.0, "provenance": "recorded"}},
           "materialisation": {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                           "n_examples_full": {"value": 10, "provenance": "derived"}, "score_kind": "binary",
                                           "load_s": {"value": None, "provenance": "absent"}, "scale": [0, 100]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/ev.json", "w"))
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 1 --occupancy /tmp/out/occupancy.jsonl > /tmp/out/daemon.log 2>&1 &
D=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python ev.py --workload gsm8k > /tmp/out/S.out 2> /tmp/out/S.err; echo $? > /tmp/out/S.rc
python3 - --workload gsm8k > /tmp/out/X.out 2> /tmp/out/X.err <<PY
import sys
path = "/tmp/work/m/ev.py"
src = open(path).read()
g = {"__name__": "__main__", "__file__": path}
sys.argv = ["ev.py"] + sys.argv[1:]
exec(compile(src, path, "exec"), g)
PY
echo $? > /tmp/out/X.rc
python3 - --workload gsm8k > /tmp/out/Y.out 2> /tmp/out/Y.err <<PY
import sys
path = "/tmp/work/m/ev.py"
src = open(path).read().replace("score_prediction", "score_one")
g = {"__name__": "__main__", "__file__": path}
sys.argv = ["ev.py"] + sys.argv[1:]
exec(compile(src, path, "exec"), g)
PY
echo $? > /tmp/out/Y.rc
kill $D 2>/dev/null; wait $D 2>/dev/null
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = 1 ]; then echo "[ok] $1"; else echo "[FAIL] $1: ${3:-}"; fails=$((fails+1)); fi; }
for t in S X Y; do echo "--- $t rc $(cat /tmp/out/$t.rc): $(grep TEST_METRICS /tmp/out/$t.out) $(grep -o "nvrt replay.*\(lookup\|seams\|withdrawn\).*" /tmp/out/$t.err | head -3 | tr "\n" " " | cut -c1-300) $(grep -E "Error|Traceback" /tmp/out/$t.err | head -2 | tr "\n" " ")"; done
chk "S script: exit 0, served, the record score 30.0" "$( [ "$(cat /tmp/out/S.rc)" = 0 ] && grep -q "final_score=30.000000" /tmp/out/S.out && echo 1 || echo 0)"
chk "X exec into a dict: exit 0, seams found in that dict, the record score 30.0, no traceback" "$( [ "$(cat /tmp/out/X.rc)" = 0 ] && grep -q "final_score=30.000000" /tmp/out/X.out && ! grep -q Traceback /tmp/out/X.err && grep -q "seams installed" /tmp/out/X.err && echo 1 || echo 0)"
chk "Y exec with the score seam renamed: exit 0, replay withdrawn, the modelled path ran (its own score 50.0), no traceback" "$( [ "$(cat /tmp/out/Y.rc)" = 0 ] && grep -q "final_score=50.000000" /tmp/out/Y.out && ! grep -q Traceback /tmp/out/Y.err && grep -q "replay withdrawn" /tmp/out/Y.err && echo 1 || echo 0)"
chk "a replay_disengaged event names per_example_seams_not_found, for Y only" "$(python -c "
import json; ev=[json.loads(l) for l in open(\"/tmp/out/occupancy.jsonl\")]
d=[e for e in ev if e[\"kind\"]==\"replay_disengaged\"]
print(1 if len(d)==1 and d[0].get(\"reason\")==\"per_example_seams_not_found\" else 0)")"
echo "$((n-fails)) passed, $fails failed"
exit $fails
'
