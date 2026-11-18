#!/bin/sh
# Start the controller gate unless it is already running. Safe to call before every run.
# State lives outside the repo and outside every container: $P006_GATE_HOME (default ~/.p006-gate).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
G=${P006_GATE_HOME:-$HOME/.p006-gate}
mkdir -p "$G/stop"
if [ ! -s "$G/link.key" ]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$G/link.key")
fi
if [ ! -s "$G/gate.json" ]; then
  : "${CW_UPSTREAM:?set CW_UPSTREAM to the model router or API the gate forwards to, e.g. http://127.0.0.1:8000}"
  printf '{"mode":"gate","upstream":"%s","log_path":"%s/gate.jsonl","stop_dir":"%s/stop","link_token_file":"%s/link.key"}\n' \
    "$CW_UPSTREAM" "$G" "$G" "$G" > "$G/gate.json"
fi
if pgrep -f "forwarder.py $G/gate.json" >/dev/null 2>&1; then
  echo "gate already running: $(pgrep -f "forwarder.py $G/gate.json" | head -1)"; exit 0
fi
if lsof -nP -iTCP:8790 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port 8790 is taken by something that is not this gate:" >&2
  lsof -nP -iTCP:8790 -sTCP:LISTEN >&2; exit 1
fi
nohup python3 -I "$HERE/forwarder.py" "$G/gate.json" >>"$G/gate.stderr.log" 2>&1 &
sleep 2
if pgrep -f "forwarder.py $G/gate.json" >/dev/null 2>&1; then
  echo "gate started: $(pgrep -f "forwarder.py $G/gate.json" | head -1), log $G/gate.jsonl"
else
  echo "gate failed to start; see $G/gate.stderr.log" >&2; tail -5 "$G/gate.stderr.log" >&2; exit 1
fi
