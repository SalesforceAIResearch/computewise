#!/usr/bin/env bash
# A failed pristine pre-import of torch._dynamo must not hide the real error.
# cpu-tasks' H4 staging check (2026-10-24) ran the filter image bare, as the
# host's uid, which has no passwd entry in the image. torch 2.11's
# torch/_dynamo/package.py registers a cache artifact at import (line 459) and
# then builds the inductor cache dir at module level (line 1173), which calls
# getpass.getuser() and raises KeyError for such a uid. The patch pre-imports
# torch._dynamo at `import torch` (pristine, before FakeGPU's patch) and used
# to swallow that failure; torch's own lazy import later re-ran package.py and
# died with "Artifact of type=precompile already registered", hiding the cause.
# cuda_hooks._preimport_dynamo now rolls back what a failed attempt registered,
# so the next import fails the way stock torch does, with the getpwuid KeyError.
# Checks, all bare (--entrypoint bash, no sidecar, --network none):
#   uid 4242, HOME=/tmp:  `import torch._dynamo` raises the KeyError, not the
#     assertion; torch.from_file (the first op under the patch's dispatch mode
#     enters dynamo's lazy import) raises the same KeyError; NVRT_DEBUG=1 names
#     the swallowed error; with TORCHINDUCTOR_CACHE_DIR set dynamo loads.
#   root and uid 1000 (the box user): dynamo fully loaded at `import torch`,
#     from_file works — boxes are unaffected.
# Usage (on the VM):  sudo IMG=<box image id> [PATCH_SRC=1] bash tests/test_dynamo_preimport_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
T=$(mktemp -d /tmp/gm-dynpre.XXXXXX); chmod 755 "$T"
cat > "$T/imp.py" <<'PY'
import torch
import torch._dynamo
print("DYNAMO_OK")
PY
cat > "$T/ff.py" <<'PY'
import torch
h = torch.from_file("/tmp/z.bin", False, 256, dtype=torch.int32)
print("FROMFILE_OK", tuple(h.shape))
PY
cat > "$T/loaded.py" <<'PY'
import sys, torch
print("LOADED", "torch._dynamo" in sys.modules, "PACKAGE", "torch._dynamo.package" in sys.modules)
PY
chmod 644 "$T"/*.py
pass=0; fail=0
ok()  { echo "[ok  ] $1"; pass=$((pass+1)); }
bad() { echo "[FAIL] $1"; fail=$((fail+1)); }
probe() {  # probe <user> <script> [docker -e args...]  -> stdout+stderr
  local user=$1 script=$2; shift 2
  docker run --rm --network none --cpus 2 --memory 4g --pids-limit 256 --entrypoint bash \
    --user "$user" -e HOME=/tmp "$@" "${M[@]}" -v "$T:/gm/t:ro" "$IMG" \
    -c "cd /tmp && head -c 2048 /dev/zero > /tmp/z.bin && python /gm/t/$script" 2>&1 | grep -v flop_counter
}
KEY='getpwuid(): uid not found'; ASSERT='already registered in mega-cache'

o=$(probe 4242:4242 imp.py)
if grep -q "$KEY" <<<"$o" && ! grep -q "$ASSERT" <<<"$o" && ! grep -q DYNAMO_OK <<<"$o"; then ok "uid 4242: import torch._dynamo raises the genuine KeyError, not the assertion"; else bad "uid 4242: import torch._dynamo -> $(tail -1 <<<"$o" | cut -c1-160)"; fi

o=$(probe 4242:4242 ff.py)
if grep -q "$KEY" <<<"$o" && ! grep -q "$ASSERT" <<<"$o"; then ok "uid 4242: from_file (first op under the dispatch mode) raises the same KeyError"; else bad "uid 4242: from_file -> $(tail -1 <<<"$o" | cut -c1-160)"; fi

o=$(probe 4242:4242 imp.py -e NVRT_DEBUG=1)
if grep -q "nvrt: dynamo pre-import failed (KeyError" <<<"$o"; then ok "uid 4242, NVRT_DEBUG=1: the swallowed pre-import error is named on stderr"; else bad "uid 4242, NVRT_DEBUG=1: no pre-import line: $(grep -m1 nvrt <<<"$o" | cut -c1-160)"; fi

o=$(probe 4242:4242 imp.py -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti)
if grep -q DYNAMO_OK <<<"$o"; then ok "uid 4242 with TORCHINDUCTOR_CACHE_DIR: dynamo imports"; else bad "uid 4242 with TORCHINDUCTOR_CACHE_DIR -> $(tail -1 <<<"$o" | cut -c1-160)"; fi

for u in 0:0 1000:1000; do
  o=$(probe $u loaded.py)
  if grep -q "LOADED True PACKAGE True" <<<"$o"; then ok "uid ${u%%:*}: torch._dynamo fully loaded at import torch"; else bad "uid ${u%%:*}: $(tail -1 <<<"$o" | cut -c1-160)"; fi
  o=$(probe $u ff.py)
  if grep -q "FROMFILE_OK (256,)" <<<"$o"; then ok "uid ${u%%:*}: from_file works"; else bad "uid ${u%%:*}: from_file -> $(tail -1 <<<"$o" | cut -c1-160)"; fi
done
rm -rf "$T"
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
