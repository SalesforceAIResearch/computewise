"""Read one run's model calls from the model router's cost ledger, read-only.

usage:
  python3 -I ledger.py dump  <cost_tag> <out.jsonl>     # copy the run's turns into the run directory
  python3 -I ledger.py spend <cost_tag>                 # print total USD so far
  python3 -I ledger.py watch <cost_tag> <cap_usd> <stop_file> [--every 15] [--budget-json <path>]
                                                       # touch stop_file once spend crosses cap
  python3 -I ledger.py tokens <cost_tag>                # print the run's input (cached included) and output tokens
  python3 -I ledger.py watch-tokens <cost_tag> <stop_file> --input <n> --output <n> [--every 15] [--json <path>]
                                                       # touch stop_file once either token sum crosses its cap

The ledger is ~/.claude/router-ledger.sqlite (override with MUX_LEDGER). It belongs to
the model router: it is opened with mode=ro and never written. `watch` is the runner's budget
enforcer: isolation's forwarder gate answers 429 for a run while <stop_dir>/<tag> exists,
so touching that file is the whole stop on the model side.

Only raw columns are copied (per turn: ts, ts_epoch, backend, model, requested_model,
session, status, stop_reason, error, stream, token counts, usd, priced, latency_ms,
queued_ms). Charged latency is computed later from these, not here.
"""
import contextlib, json, os, sqlite3, sys, time

COLS = ["id", "ts", "ts_epoch", "backend", "model", "requested_model", "effort", "tag", "tag_src",
        "session", "status", "stop_reason", "error", "stream", "input_tokens", "cache_read",
        "cache_write", "output_tokens", "usd", "upstream_usd", "priced", "latency_ms", "queued_ms", "agent_id",
        "parent_agent_id", "routed_from", "billing"]

# a watcher ends when its run does, read from the campaign's claim for its tag (the study 2026-11-02 20:08Z: ten token watchers of
# finished runs, 1-4 days old, held ~300% CPU, because nothing wrote <stop>.done when a run ended): the claim whose tag is the
# watcher's, once its status is no longer running or it has been collected. P006_CLAIMS overrides the directory (tests)
CLAIMS = os.environ.get("P006_CLAIMS") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "campaign", "claims"))
_CLAIM_OF = {}
def run_ended(tag):
    """True when the claim carrying this tag has left running or was collected; False while it runs, or with no claim found."""
    path = _CLAIM_OF.get(tag)
    if path is None:
        try:
            names = os.listdir(CLAIMS)
        except OSError:
            return False
        for f in names:
            if f.endswith(".json"):
                try:
                    c = json.load(open(os.path.join(CLAIMS, f)))
                except (OSError, ValueError):
                    continue
                if c.get("tag") == tag:
                    path = _CLAIM_OF[tag] = os.path.join(CLAIMS, f)
                    break
    if path is None:
        return False
    try:
        c = json.load(open(path))
    except (OSError, ValueError):
        return False
    return c.get("tag") == tag and (bool(c.get("collected_utc")) or c.get("status") not in ("running", "claimed", "starting", None))

def connect():
    """The ledger, read-only, closed when the with-block ends. Closed explicitly, because a watcher calls this every 15 s for the
    length of its run and an unclosed connection kept ~2.5 MB a call (2026-11-04: G-k2lp-astra-L0-v2-r2's watch-tokens at 2.2 GB
    after 9 h, on a controller out of swap); sqlite3's own with-block commits and does not close"""
    p = os.environ.get("CW_LEDGER") or sys.exit("ledger: set $CW_LEDGER to the model router's SQLite cost ledger")
    return contextlib.closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10))

def turns(tag):
    with connect() as c:
        have = {r[1] for r in c.execute("pragma table_info(turns)")}
        cols = [x for x in COLS if x in have]
        rows = c.execute(f"select {', '.join(cols)} from turns where tag = ? order by id", (tag,)).fetchall()
    return [dict(zip(cols, r)) for r in rows]

def spend(tag):
    # the same figure budget_watch.py enforces the ceiling on: the upstream cost where the router recorded
    # one, else its own price (COALESCE(upstream_usd, usd), the cost the the router dashboard shows)
    with connect() as c:
        usd, n, unpriced = c.execute("select coalesce(sum(coalesce(upstream_usd, usd)),0), count(*), coalesce(sum(priced=0),0) "
                                     "from turns where tag = ?", (tag,)).fetchone()
    return float(usd), int(n), int(unpriced)

def tokens(tag):
    """The run's token sums: input counts every prompt token the model read, cached ones included (input_tokens +
    cache_read + cache_write), and output its generated tokens. A token ceiling for a model the dollar figure does not
    bound well, enforced like the dollar one: the stop file."""
    with connect() as c:
        i, o, n = c.execute("select coalesce(sum(coalesce(input_tokens,0) + coalesce(cache_read,0) + coalesce(cache_write,0)),0), "
                            "coalesce(sum(coalesce(output_tokens,0)),0), count(*) from turns where tag = ?", (tag,)).fetchone()
    return int(i), int(o), int(n)

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, tag = sys.argv[1], sys.argv[2]
    if cmd == "dump":
        out = sys.argv[3]
        rows = turns(tag)
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        usd = sum((r.get("upstream_usd") if r.get("upstream_usd") is not None else r.get("usd")) or 0 for r in rows)
        print(f"{len(rows)} turns, ${usd:.4f}, backends {sorted({r['backend'] for r in rows})} -> {out}")
    elif cmd == "spend":
        usd, n, unpriced = spend(tag)
        print(f"{tag}: {n} turns, ${usd:.4f}" + (f", {unpriced} unpriced" if unpriced else ""))
    elif cmd == "watch":
        cap, stop = float(sys.argv[3]), sys.argv[4]
        every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 15.0
        bj = sys.argv[sys.argv.index("--budget-json") + 1] if "--budget-json" in sys.argv else None
        while True:
            usd, n, unpriced = spend(tag)
            if usd >= cap:
                crossed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                os.makedirs(os.path.dirname(stop) or ".", exist_ok=True)
                open(stop, "a").close()
                rec = {"cap_usd": cap, "spent_usd": usd, "turns": n, "unpriced_turns": unpriced,
                       "crossed_utc": crossed, "stop_touched_utc": crossed, "stop_file": stop}
                if bj:
                    json.dump(rec, open(bj, "w"), indent=1)
                print(json.dumps(rec)); return
            if os.path.exists(stop + ".done") or run_ended(tag):
                return
            time.sleep(every)
    elif cmd == "tokens":
        i, o, n = tokens(tag)
        print(f"{tag}: {n} turns, {i} input tokens (cached included), {o} output tokens")
    elif cmd == "watch-tokens":
        stop = sys.argv[3]
        cap_in = float(sys.argv[sys.argv.index("--input") + 1])
        cap_out = float(sys.argv[sys.argv.index("--output") + 1])
        every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 15.0
        jp = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
        while True:
            i, o, n = tokens(tag)
            if i >= cap_in or o >= cap_out:
                crossed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                os.makedirs(os.path.dirname(stop) or ".", exist_ok=True)
                rec = {"cap_input_tokens": cap_in, "cap_output_tokens": cap_out, "input_tokens": i, "output_tokens": o,
                       "turns": n, "crossed": "input" if i >= cap_in else "output", "crossed_utc": crossed,
                       "stop_touched_utc": crossed, "stop_file": stop}
                # the stop file names its writer, so the dollar watcher can tell a token stop from its own and never lift
                # it (the study, 2026-10-22: budget_watch removed G-goalq-astra-L0-v2's token stop 5 s after it was
                # written, and the run went on 17 min and $1 past its cap). Written whole and renamed into place
                tmp = stop + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"writer": "watch-tokens", "sticky": True, **rec}, fh)
                os.replace(tmp, stop)
                if jp:
                    json.dump(rec, open(jp, "w"), indent=1)
                print(json.dumps(rec)); return
            if os.path.exists(stop + ".done") or os.path.exists(stop) or run_ended(tag):
                return              # the run ended, or another watcher (the dollar one) already stopped it
            time.sleep(every)
    else:
        sys.exit(__doc__)

if __name__ == "__main__":
    main()
