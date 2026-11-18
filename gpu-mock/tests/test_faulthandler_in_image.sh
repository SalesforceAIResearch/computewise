#!/usr/bin/env bash
# faulthandler dumps inside the box must not show the patch's frames.
# G-h1-haiku-L0 (2026-10-23) ran a hung smoke test under
# `faulthandler.dump_traceback_later(90, file=/tmp/fh.txt)`, read
# `/usr/local/nvidia/python/flop_charger.py ... in __torch_dispatch__` in the
# dump, then read that file. faulthandler writes C-level frames to a file
# descriptor, so the traceback-module filtering never sees them; sitecustomize
# now routes dump_traceback / dump_traceback_later / register through a
# Python-level writer in faulthandler's format with the patch's frames left out.
# Checks: a timed dump during a loop of cuda matmuls (every op runs inside the
# patch's dispatch) exits the process as asked, carries the Timeout header and
# the script's frame, and no /usr/local/nvidia frame; the same under
# NVRT_DEBUG=1 (native faulthandler) DOES show the patch frame, which makes the
# first check conclusive; a direct dump_traceback() keeps the native format.
# Usage (on the VM):  sudo IMG=<box image id> [PATCH_SRC=1] bash tests/test_faulthandler_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 2 --memory 4g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
cd /tmp
cat > spin.py <<PY
import faulthandler, sys, torch
mode = sys.argv[1]
f = open("/tmp/fh-%s.txt" % mode, "w")
faulthandler.dump_traceback_later(2, file=f, exit=True)
x = torch.randn(1024, 1024, device="cuda")
while True:
    y = x @ x
PY
timeout 60 python spin.py filtered; rc1=$?
NVRT_DEBUG=1 timeout 60 python spin.py control; rc2=$?
python - <<PY > /tmp/fh-direct.txt 2>&1
import faulthandler, sys
def inner():
    faulthandler.dump_traceback(file=sys.stdout, all_threads=False)
inner()
PY
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = 1 ]; then echo "[ok] $1"; else echo "[FAIL] $1: ${3:-}"; fails=$((fails+1)); fi; }
chk "1: the timed dump exited the process (exit=True honoured)" "$([ $rc1 = 1 ] && echo 1 || echo 0)" "rc=$rc1"
chk "1: the dump carries the Timeout header and the script frame" "$( (grep -q "^Timeout (0:00:02)!" /tmp/fh-filtered.txt && grep -q "spin.py\", line [0-9]* in <module>" /tmp/fh-filtered.txt) && echo 1 || echo 0)" "$(head -4 /tmp/fh-filtered.txt | tr "\n" "|")"
chk "1: no patch frame in the dump" "$(! grep -q "/usr/local/nvidia" /tmp/fh-filtered.txt && echo 1 || echo 0)" "$(grep nvidia /tmp/fh-filtered.txt | head -2 | tr "\n" "|")"
if grep -q "/usr/local/nvidia" /tmp/fh-control.txt; then chk "control (NVRT_DEBUG=1, native faulthandler) shows the patch frame: check 1 is conclusive" 1; else echo "[inconclusive] the native control dump did not land inside a patch frame this time: $(head -3 /tmp/fh-control.txt | tr "\n" "|")"; fi
chk "1: no writer frame and no Current-thread label in the timed dump (a watchdog dump has neither), main thread block present" "$( (! grep -q "sitecustomize.py" /tmp/fh-filtered.txt && ! grep -q "^Current thread" /tmp/fh-filtered.txt && grep -q "^Thread 0x" /tmp/fh-filtered.txt) && echo 1 || echo 0)" "$(head -5 /tmp/fh-filtered.txt | tr "\n" "|")"
chk "2: a direct dump_traceback() keeps the thread header and the caller frames, no writer frame" "$( (grep -q "^Current thread 0x" /tmp/fh-direct.txt && grep -q "in inner" /tmp/fh-direct.txt && ! grep -q "sitecustomize.py" /tmp/fh-direct.txt) && echo 1 || echo 0)" "$(head -3 /tmp/fh-direct.txt | tr "\n" "|")"
echo "=== native control dump (head, NVRT_DEBUG=1)"; head -8 /tmp/fh-control.txt
echo "=== filtered dump (all)"; cat /tmp/fh-filtered.txt
echo "=== direct dump"; cat /tmp/fh-direct.txt
echo "$((n-fails)) passed, $fails failed"
exit $fails
'
