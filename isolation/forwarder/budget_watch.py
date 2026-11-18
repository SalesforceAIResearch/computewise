#!/usr/bin/env python3
"""Per-run USD ceilings for the gate, read from the model router's cost ledger (an SQLite file).

Every interval_s it sums each run's spend in the ledger (table turns: tag, month, usd, upstream_usd; COALESCE(upstream_usd,
usd)) and creates <stop_dir>/<run tag> once a run reaches its
ceiling. The gate answers that run's requests with 429 while the file exists. Raising the ceiling
lifts a stop this watcher wrote, at the next check; deleting the file lifts any stop.

It lifts only its own stops: a file it wrote ({"writer": "budget_watch", "run", "usd", "ceiling", ...}) and that
does not say "sticky": true. Any other stop file stays, and the refusal is logged once per file as
{"event": "resume-refused", "writer": ...}: the runner's token-cap stop (ledger.py watch-tokens, sticky), an empty
file from an older writer, or one a person touched (the study's ruling, 2026-10-22 03:47Z: a stop is sticky;
G-goalq-astra-L0-v2's token stop at 02:36:26Z was removed by this watcher 5 s later and the run went on 17 min).

The ledger records a turn when it finishes, so the spend of requests still in flight is not
counted yet. A run can therefore overshoot its ceiling by the cost of what it had in flight
when it crossed, plus up to one interval of new requests.

The config is re-read on every check, so the runner can add runs while this is running (write
it to a temporary file and rename it over the old one; a half-written file is skipped):
  {"stop_dir": "/abs/stop", "interval_s": 15, "ceilings": {"P006-r0042": 2.50}}
Required key "ledger" (the SQLite path, or $CW_LEDGER); optional "log_path" (default budget.jsonl).

Run:  python3 -I budget_watch.py <config.json>          (stdlib only; opens the ledger read-only)
"""
import json
import os
import re
import sqlite3
import sys
import time

TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
REFUSED = set()   # stop files whose resume this watcher refused, so the refusal is logged once, not every interval


def months_back(n=2):
    """The current local month and the n-1 before it, as the ledger's 'YYYY-MM' keys (index use)."""
    y, m = time.localtime()[:2]
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out


def spend(conn, tag, months):
    q = ("SELECT COALESCE(SUM(COALESCE(upstream_usd, usd)), 0), COUNT(*) FROM turns "
         f"WHERE tag = ? AND month IN ({','.join('?' * len(months))})")
    return conn.execute(q, (tag, *months)).fetchone()


def check(cfg, log):
    ledger = os.path.expanduser(cfg.get("ledger") or os.environ.get("CW_LEDGER") or sys.exit("budget_watch: set \"ledger\" in the config or $CW_LEDGER"))
    stop_dir = cfg["stop_dir"]
    os.makedirs(stop_dir, exist_ok=True)
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=10)
    try:
        months = months_back()
        for tag, ceiling in (cfg.get("ceilings") or {}).items():
            if not TAG_RE.match(tag):
                log({"event": "bad_tag", "run": tag})
                continue
            usd, turns = spend(conn, tag, months)
            flag = os.path.join(stop_dir, tag)
            over = usd >= float(ceiling)
            if over and not os.path.exists(flag):
                tmp = f"{flag}.bw{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump({"writer": "budget_watch", "run": tag, "usd": round(usd, 4), "ceiling": ceiling,
                               "turns": turns, "ts_epoch": round(time.time(), 3)}, f)
                os.replace(tmp, flag)
                log({"event": "stop", "run": tag, "usd": round(usd, 4), "ceiling": ceiling, "turns": turns})
            elif not over and os.path.exists(flag):
                body = {}
                try:
                    with open(flag) as f:
                        body = json.load(f)
                except (OSError, ValueError):
                    pass
                if not isinstance(body, dict):
                    body = {}
                # Ours: written by this watcher (the "writer" key, or an older copy of it, which wrote run, usd and
                # ceiling and no writer), for this run, and not marked sticky by anyone.
                ours = (body.get("writer") in (None, "budget_watch") and not body.get("sticky")
                        and "ceiling" in body and body.get("run") == tag)
                if ours:
                    os.remove(flag)
                    REFUSED.discard(flag)
                    log({"event": "resume", "run": tag, "usd": round(usd, 4), "ceiling": ceiling, "turns": turns})
                elif flag not in REFUSED:
                    REFUSED.add(flag)
                    log({"event": "resume-refused", "run": tag, "writer": body.get("writer") or "unknown",
                         "sticky": bool(body.get("sticky")), "usd": round(usd, 4), "ceiling": ceiling})
    finally:
        conn.close()


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: budget_watch.py <config.json>")
    path = sys.argv[1]
    log_path = None

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps({"ts_epoch": round(time.time(), 3), **rec}, separators=(",", ":")) + "\n")

    with open(path) as f:
        cfg = json.load(f)                      # a bad config at start is fatal
    while True:
        try:
            with open(path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):           # mid-rewrite; keep the last good config
            pass
        log_path = cfg.get("log_path") or "budget.jsonl"
        try:
            check(cfg, log)
        except sqlite3.Error as exc:            # the ledger is busy or rotating; try next interval
            log({"event": "ledger_error", "error": str(exc)})
        time.sleep(float(cfg.get("interval_s", 15)))


if __name__ == "__main__":
    main()
