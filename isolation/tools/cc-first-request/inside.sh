#!/bin/sh
# Runs inside the box image with no network. One Claude Code invocation per variant, each against a fresh
# local stub on port 8080. The base URL is the real run's (api-gw.internal) when the container maps that
# name to 127.0.0.1, and 127.0.0.1 otherwise; meta.txt says which.
set -u
export HOME=/tmp/h USER=researcher LOGNAME=researcher SHELL=/bin/bash
if getent hosts api-gw.internal >/dev/null 2>&1; then BASE=http://api-gw.internal:8080; else BASE=http://127.0.0.1:8080; fi
export ANTHROPIC_BASE_URL=$BASE
export ANTHROPIC_API_KEY=sk-ant-api03-0000000000000000000000000000000000000000
export ANTHROPIC_MODEL=claude-haiku-5-5 ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-5-5 \
       ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-5-5 ANTHROPIC_DEFAULT_SONNET_MODEL=claude-haiku-5-5 \
       ANTHROPIC_DEFAULT_OPUS_MODEL=claude-haiku-5-5 ANTHROPIC_DEFAULT_FABLE_MODEL=claude-haiku-5-5 \
       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 \
       BASH_DEFAULT_TIMEOUT_MS=36000000 BASH_MAX_TIMEOUT_MS=36000000
CL=$(command -v claude)
{ echo "claude: $CL"; "$CL" --version 2>&1 | head -1; echo "base_url: $BASE"; } > /out/meta.txt
for v in ${VARIANTS:-default updates_0 updates_false flag_omitted}; do
  rm -rf "$HOME" /tmp/p; mkdir -p "$HOME" /tmp/p; cd /tmp/p
  python3 -I /t/stub.py --port 8080 --out /out/$v.jsonl & sp=$!
  sleep 1
  ENVS="VARIANT=$v"; ARGS=""
  case $v in
    updates_0) ENVS="$ENVS CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0" ;;
    updates_false) ENVS="$ENVS CLAUDE_CODE_THINKING_DISPLAY_UPDATES=false" ;;
    flag_omitted) ARGS="--thinking-display omitted" ;;
    summarized_0) ENVS="$ENVS CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0"; ARGS="--thinking-display summarized" ;;
    summarized) ARGS="--thinking-display summarized" ;;
  esac
  s=$(date +%s)
  # ENVS and ARGS hold no spaces inside a value, so word splitting is intended here.
  env $ENVS timeout 120 "$CL" -p --output-format stream-json --verbose --permission-mode bypassPermissions \
      --model claude-haiku-5-5 --session-id "$(cat /proc/sys/kernel/random/uuid)" $ARGS \
      "Reply with the single word ok." < /dev/null > /out/$v.stream 2> /out/$v.stderr
  rc=$?
  echo "$v exit=$rc wall_s=$(( $(date +%s) - s ))" >> /out/meta.txt
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
done
exit 0
