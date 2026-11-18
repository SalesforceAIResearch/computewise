#!/usr/bin/env bash
# The box as built carries the torch layer without our prose (G-kvgate-astracc-L0-r2,
# 2026-10-27: an Explore subagent grepped site-packages for `cuda|version`, read
# sitecustomize.py, and its first line said "Loads the gpu-mock torch layer").
# Reads the INSTALLED files of an image built from this tree, the way an agent would:
#   1. site-packages/sitecustomize.py, /usr/local/nvidia/python/*.py and
#      /usr/local/nvidia/pylib/nvcompat/**/*.py hold no comment and no docstring;
#   2. sitecustomize.py and the torch layer name none of: gpu-mock, mock, fake,
#      simulat, emulat, the project or its sessions (FakeGPU's own API names, which
#      the torch layer calls by name, excepted and counted);
#   3. torch imports, the patch applies, and a cuda matmul runs;
#   4. the agent's own grep (`cuda|version` in site-packages) lists sitecustomize.py,
#      and its first 5 lines are code.
# Usage (on the VM):  sudo IMG=sha256:<box id> bash tests/test_installed_text_in_image.sh
set -uo pipefail
IMG="${IMG:?box image id}"
docker run --entrypoint '' --rm --cpus 2 --memory 4g "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; cd /tmp
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = 1 ]; then echo "[ok] $1"; else echo "[FAIL] $1: ${3:-}"; fails=$((fails+1)); fi; }
python3 - "$SITE/sitecustomize.py" /usr/local/nvidia/python /usr/local/nvidia/pylib/nvcompat > /tmp/scan.txt <<PY
import ast, io, os, re, sys, tokenize
words = re.compile(r"gpu[-_ ]?mock|\bmock|fake|simulat|emulat|ComputeWise|printing.press|gpu-tasks|cpu-tasks|the study|the analysis", re.I)
fg_api = re.compile(r"\b_?Fake(?:CudaTensor|Stream|Event|DeviceProperties|\{attr\})\b|_Fake\{attr\}|\bFAKEGPU_[A-Z_]+|[\x22\x27]fakegpu[\x22\x27.]|fakegpu\.\{sub\}|startswith\([\x22\x27]fakegpu[\x22\x27]\)|FakeGPU has")
files = [sys.argv[1]]
for d in sys.argv[2:]:
    for root, _, names in os.walk(d):
        files += [os.path.join(root, x) for x in names if x.endswith(".py")]
ours = [0, 0, set(), 0]; fg = [0, 0]
for p in files:
    src = open(p).read()
    com = sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT)
    doc = sum(1 for node in ast.walk(ast.parse(src)) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False))
    if "/pylib/nvcompat/" in p:
        fg[0] += com; fg[1] += doc
    else:
        ours[0] += com; ours[1] += doc; ours[3] += len(fg_api.findall(src))
        ours[2].update(m.group(0).lower() for m in words.finditer(fg_api.sub(" ", src)))
print("files", len(files))
print("ours comments", ours[0], "docstrings", ours[1], "fakegpu_api", ours[3], "words", ",".join(sorted(ours[2])))
print("nvcompat comments", fg[0], "docstrings", fg[1])
PY
cat /tmp/scan.txt
O=$(grep "^ours" /tmp/scan.txt); V=$(grep "^nvcompat" /tmp/scan.txt)
chk "1: no comment and no docstring in sitecustomize.py, the torch layer or nvcompat" "$(python3 -c "import re,sys; n=[int(x) for x in re.findall(r\"(?:comments|docstrings) (\d+)\", sys.argv[1])]; print(1 if len(n) == 4 and not any(n) else 0)" "$O $V")" "$O | $V"
chk "2: sitecustomize.py and the torch layer name nothing of the mock or the project (FakeGPU API names excepted)" "$(python3 -c "import sys; print(1 if not sys.argv[1].split(\"words\", 1)[1].strip() else 0)" "$O")" "$O"
python3 -c "import torch; x = torch.randn(64, 64, device=\"cuda\"); print(\"CUDA_OK\", x.is_cuda, float((x @ x).sum()) == float((x @ x).sum()))" > /tmp/imp.txt 2>&1
chk "3: torch imports, the patch applies, a cuda matmul runs" "$(grep -q "^CUDA_OK True True" /tmp/imp.txt && echo 1 || echo 0)" "$(tail -3 /tmp/imp.txt)"
G=$(grep -rlE "cuda|version" "$SITE" --include=sitecustomize.py 2>/dev/null | head -1)
chk "4: the agent grep finds sitecustomize.py, and its first 5 lines are code" "$( [ -n "$G" ] && ! head -5 "$G" | grep -qE "^\s*(#|\x22\x22\x22)" && echo 1 || echo 0)" "$(head -5 "$G" | tr "\n" "|")"
echo "--- head of the installed sitecustomize.py:"; head -6 "$SITE/sitecustomize.py"
echo "$((n-fails)) passed, $fails failed"
exit $fails
'
