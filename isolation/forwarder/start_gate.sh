#!/bin/sh
# Start the controller gate unless it is already running. Safe to call before every run.
# State lives outside the repo and outside every container: $CW_GATE_HOME (default /srv/stage/gate, where run_one.py
# looks for the link key and the stop directory by default; pass run_one --link-key and --stop-dir for another place).
# The gate listens on Docker's bridge gateway (CW_GATE_LISTEN to override), where each run's edge container reaches it.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
G=${CW_GATE_HOME:-${P006_GATE_HOME:-/srv/stage/gate}}
LISTEN=${CW_GATE_LISTEN:-$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || echo 127.0.0.1)}
mkdir -p "$G/stop"
if [ ! -s "$G/link.key" ]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$G/link.key")
fi
if [ ! -s "$G/gate.json" ]; then
  : "${CW_UPSTREAM:?set CW_UPSTREAM to the model router or API the gate forwards to, e.g. https://api.anthropic.com}"
  # CW_UPSTREAM_KEY_ENV: the name of the variable holding your API key (e.g. ANTHROPIC_API_KEY), when the upstream is
  # the provider's API; the gate sends it upstream. Leave it unset for a local router that adds its own credentials.
  # CW_UPSTREAM_KEY_HEADER: x-api-key (default, Anthropic) or authorization (sent as Bearer <key>).
  KEYCFG=""
  if [ -n "${CW_UPSTREAM_KEY_ENV:-}" ]; then
    KEYCFG=$(printf ',"upstream_key_env":"%s","upstream_key_header":"%s"' "$CW_UPSTREAM_KEY_ENV" "${CW_UPSTREAM_KEY_HEADER:-x-api-key}")
  fi
  printf '{"mode":"gate","listen_host":"%s","upstream":"%s","log_path":"%s/gate.jsonl","stop_dir":"%s/stop","link_token_file":"%s/link.key"%s}\n' \
    "$LISTEN" "$CW_UPSTREAM" "$G" "$G" "$G" "$KEYCFG" > "$G/gate.json"
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
