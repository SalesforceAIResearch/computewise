#!/bin/bash
# Check a data-preparation package's raw text on the VM: decode the kjj0 shards to documents with
# build/gen_raw_text.py, tokenise them with the reference preparation (artifacts/mng3/dataprep/fineweb_local.py, data/fineweb.py
# reading data/raw/), and compare every shard it writes with the original by sha256.
# usage: check_dataprep.sh <gpu-tasks dir on the VM> <kjj0 shard dir> <work dir>
set -euo pipefail
G=$1; S=$2; W=$3
mkdir -p "$W"; cd "$W"
export PATH=$HOME/.local/bin:$PATH
if [ ! -x venv/bin/python ]; then
  uv venv -q --python 3.11.9 venv
  uv pip install -q --python venv/bin/python numpy==2.4.4 tiktoken tqdm
fi
venv/bin/python -c "import tiktoken, numpy, tqdm; print('tiktoken', tiktoken.__version__, 'numpy', numpy.__version__, 'tqdm', tqdm.__version__)"
REV=$(curl -sf https://huggingface.co/api/models/openai-community/gpt2 | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
[ -f vocab.json ] || curl -sfL -o vocab.json "https://huggingface.co/openai-community/gpt2/resolve/$REV/vocab.json"
echo "gpt2 vocab.json at openai-community/gpt2@$REV sha256 $(sha256sum vocab.json | cut -d' ' -f1)"
rm -rf tree; mkdir -p tree/data
t0=$(date +%s.%N)
python3 -I "$G/build/gen_raw_text.py" "$S" vocab.json tree/data/raw | tee raw.txt
t1=$(date +%s.%N); echo "gen_raw_text: $(echo "$t1 - $t0" | bc) s"
du -sb tree/data/raw
cp "$G/registry/artifacts/mng3/dataprep/fineweb_local.py" tree/data/fineweb_local.py
t0=$(date +%s.%N)
(cd tree && ../venv/bin/python data/fineweb_local.py > ../tok.log 2>&1)
t1=$(date +%s.%N); echo "fineweb_local ($(nproc) cpus, $(( $(nproc) - 2 )) workers): $(echo "$t1 - $t0" | bc) s"
grep -a "writing" tok.log
for f in tree/data/fineweb10B/*.bin; do
  b=$(basename "$f"); a=$(sha256sum "$f" | cut -d' ' -f1)
  if [ -f "$S/$b" ]; then o=$(sha256sum "$S/$b" | cut -d' ' -f1); else o=none; fi
  [ "$a" = "$o" ] && echo "$b IDENTICAL $a" || echo "$b DIFFERENT ($(stat -c %s "$f") bytes; original $(stat -c %s "$S/$b" 2>/dev/null || echo -) bytes)"
done
