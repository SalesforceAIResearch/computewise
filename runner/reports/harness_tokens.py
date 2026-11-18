"""What each arm costs in model tokens, beside the GPU efficiency it buys (the analysis 2026-10-29 ~08:55 MDT, for the
the study's harness trade-off: the harness spends the study tokens to use the GPUs better, against Astra in Codex ticking every
30 s). One row per valid run on G4 (mls-kv-gate), G1 (mls-kv-report) and twin A (pkg-g4n-a) in the arms L0 (Haiku, Sonnet, Opus,
Astra in Codex, Astra in Claude Code, Sol), V3, T4 and every A rung.

Per run, from the router's ledger (~/.claude/router-ledger.sqlite, opened mode=ro, rows matched by the run's cost tag):
  requests (all rows) and failed (status not 200); input, cache_read, cache_write, output and their sum; uncached =
  input + cache_write + output; the same split between main (rows with no agent_id, on the run's main provider thread) and
  other (teammates and subagents: rows with an agent_id, or on another Codex thread); main's peak context (the largest
  input + cache_read + cache_write of one main row); tokens and requests per hour of the run (agent start to harness end,
  from status.json); and the run's bar figure in its package's column (screen.py's rows: over_ref for goal-only packages,
  over_feasible where the package has a feasible best, else over_best), with no_bar_reading and the disclosure dagger.
A run whose controller ledger copy (ledger.jsonl) holds a different row count from the sqlite is flagged in `ledger_copy_rows`.

usage: python3 -I harness_tokens.py [--out-dir DIR]     writes harness-tokens.md, .json and .csv beside this script
One process, read-only; it scores nothing."""
import argparse, csv, json, os, sqlite3, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "score"))
import screen as S                                      # noqa: E402  (the bar figures and arm parsing the screen uses)

LEDGER = os.environ.get("CW_LEDGER", "")
TASKS = {"mls-kv-gate": "G4", "mls-kv-report": "G1", "pkg-g4n-a": "twin A"}
L0_MODELS = {"haiku", "sonnet", "opus", "astra", "astra-cc", "sol"}
ORDER_ARM = {"L0": 0, "V3": 1, "T4": 2}

def wanted(row):
    arm, mod = row["arm"], row["model"]
    if arm == "L0":
        return mod in L0_MODELS
    return arm in ("V3", "T4") or arm.startswith("A")

def bar_of(row):
    t = row["task"]
    col = "over_ref" if t in S.GOAL_ONLY else "over_feasible" if S.feasible_best(t) is not None else "over_best"
    return col, row.get(col)

def months(tag):
    ym = tag.rsplit("-", 1)[-1][:6]
    y, m = int(ym[:4]), int(ym[4:6])
    nxt = f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}"
    return (f"{y}-{m:02d}", nxt)

def tokens(con, tag):
    rows = con.execute("SELECT status, input_tokens, cache_read, cache_write, output_tokens, agent_id, provider_thread_id "
                       "FROM turns WHERE month IN (?, ?) AND tag = ?", (*months(tag), tag)).fetchall()
    threads = {}
    for r in rows:
        if r[6]:
            threads[r[6]] = threads.get(r[6], 0) + 1
    main_thread = max(threads, key=threads.get) if threads else None
    z = lambda: {"requests": 0, "input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    out = {"main": z(), "other": z()}
    failed, peak = 0, 0
    for st, i, cr, cw, o, aid, th in rows:
        i, cr, cw, o = (i or 0), (cr or 0), (cw or 0), (o or 0)
        side = "other" if (aid or (main_thread and th and th != main_thread)) else "main"
        b = out[side]
        b["requests"] += 1; b["input"] += i; b["cache_read"] += cr; b["cache_write"] += cw; b["output"] += o
        if st != 200:
            failed += 1
        if side == "main":
            peak = max(peak, i + cr + cw)
    for b in out.values():
        b["total"] = b["input"] + b["cache_read"] + b["cache_write"] + b["output"]
        b["uncached"] = b["input"] + b["cache_write"] + b["output"]
    all_ = {k: out["main"][k] + out["other"][k] for k in out["main"]}
    return all_, out["main"], out["other"], failed, peak, len(rows)

def run_hours(rid):
    try:
        st = json.load(open(os.path.join(S.RUNS, rid, "status.json")))
        a, b = float(st["agent_start_wall"]), float(st.get("harness_end_wall") or st.get("teardown_wall"))
        return (b - a) / 3600.0 if b > a else None
    except (OSError, ValueError, KeyError, TypeError):
        return None

def copy_rows(rid):
    p = os.path.join(S.RUNS, rid, "ledger.jsonl")
    if not os.path.exists(p):
        return None
    return sum(1 for l in open(p) if l.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    out = []
    for r in S.rows(tasks=set(TASKS)):
        if not wanted(r):
            continue
        rid = r["run_id"]
        try:
            tag = json.load(open(os.path.join(S.RUNS, rid, "run.json")))["cost_tag"]
        except (OSError, ValueError, KeyError):
            tag = None
        col, bar = bar_of(r)
        rec = {"run_id": rid, "package": TASKS[r["task"]], "task": r["task"], "model": r["model"], "arm": r["arm"], "tag": tag,
               "bar_column": col, "bar": bar, "no_bar_reading": r["no_bar_reading"], "disclosed": r["disclosed"]}
        if tag:
            all_, m, o, failed, peak, n = tokens(con, tag)
            h = run_hours(rid)
            cp = copy_rows(rid)
            rec.update(all=all_, main=m, other=o, failed=failed, main_peak_context=peak, run_h=round(h, 3) if h else None,
                       tokens_per_h=round(all_["total"] / h) if h else None, uncached_per_h=round(all_["uncached"] / h) if h else None,
                       requests_per_h=round(all_["requests"] / h, 1) if h else None,
                       ledger_copy_rows=None if cp in (None, n) else f"{cp} in ledger.jsonl vs {n} in the sqlite")
        out.append(rec)
    out.sort(key=lambda x: (list(TASKS.values()).index(x["package"]), ORDER_ARM.get(x["arm"], 3), x["arm"], x["model"], x["run_id"]))
    json.dump(out, open(os.path.join(a.out_dir, "harness-tokens.json"), "w"), indent=1)
    flat = ["run_id", "package", "model", "arm", "bar_column", "bar", "no_bar_reading", "run_h", "failed", "main_peak_context",
            "tokens_per_h", "uncached_per_h", "requests_per_h"]
    with open(os.path.join(a.out_dir, "harness-tokens.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(flat + [f"{s}_{k}" for s in ("all", "main", "other") for k in ("requests", "input", "cache_read", "cache_write", "output", "total", "uncached")])
        for x in out:
            w.writerow([x.get(k) for k in flat] + [(x.get(s) or {}).get(k) for s in ("all", "main", "other")
                                                   for k in ("requests", "input", "cache_read", "cache_write", "output", "total", "uncached")])
    M = lambda v: "" if v is None else f"{v / 1e6:.2f}"
    K = lambda v: "" if v is None else f"{v / 1e3:.0f}k"
    def B(x):
        if x["bar"] is None:
            return "n/a"
        s = f"{x['bar']:.3f}" + ("†" if x["disclosed"] else "")
        return f"({s}) nr" if x["no_bar_reading"] else s
    L = ["# Tokens against GPU efficiency, per run", "",
         "Written by `harness_tokens.py` beside this file (the runner, for the analysis's 2026-10-29 ~08:55 MDT request). "
         "Regenerate it with `python3 -I harness_tokens.py`; the JSON and CSV beside it carry every column.", "",
         "**What each column is.**",
         "- **Source:** the router's cost ledger (sqlite, read-only, $CW_LEDGER), every row with the run's cost tag.",
         "- **Tokens:** in millions.",
         "- **uncached:** input + cache_write + output.",
         "- **main and other:** main is the root session; other is teammates and subagents (rows with an agent_id, or on another Codex thread). "
         "Codex runs show one thread each in the ledger, so they are all main.",
         "- **peak ctx:** main's largest single-request context (input + cache_read + cache_write).",
         "- **per hour:** over the run's agent start to harness end.",
         "- **bar:** the package's column: over_ref for G4 and twin A, over_best for G1. "
         "`nr` marks no_bar_reading, so the figure is shown in brackets and is not a reading; † marks a disclosed detection.",
         "- **Runs included:** valid runs only, in the arms L0 (Haiku, Sonnet, Opus, Astra in Codex as `astra`, Astra in Claude Code as `astra-cc`, Sol), V3, T4 and A*.",
         "- **Cell medians:** the bar median is over the runs with a reading; the token medians are over every valid run in the cell, `nr` included.",
         "- **Codex totals are mostly cache_read:** it resends the whole context each request at cached rates, so compare arms on **uncached** as well as total.", ""]
    for pkg in TASKS.values():
        rs = [x for x in out if x["package"] == pkg and x.get("all")]
        if not rs:
            continue
        col = rs[0]["bar_column"]
        L += [f"## {pkg} ({rs[0]['task']}, bar = {col})", "",
              "| run | model | arm | bar | h | req | req/h | input | cache_read | cache_write | output | total | uncached | main total | main uncached | other total | other uncached | peak ctx | total/h | uncached/h |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for x in rs:
            A_, m, o = x["all"], x["main"], x["other"]
            L.append(f"| {x['run_id']} | {x['model']} | {x['arm']} | {B(x)} | {x['run_h'] or ''} | {A_['requests']} | {x['requests_per_h'] or ''} | "
                     f"{M(A_['input'])} | {M(A_['cache_read'])} | {M(A_['cache_write'])} | {M(A_['output'])} | {M(A_['total'])} | {M(A_['uncached'])} | "
                     f"{M(m['total'])} | {M(m['uncached'])} | {M(o['total'])} | {M(o['uncached'])} | {K(x['main_peak_context'])} | "
                     f"{M(x['tokens_per_h'])} | {M(x['uncached_per_h'])} |")
        L.append("")
        L += [f"**{pkg}, cell medians** (model × arm; the bar median is over the runs with a reading)", "",
              "| model | arm | n | bar median | total median | uncached median | total/h median | uncached/h median | req/h median | peak ctx median |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        cells = {}
        for x in rs:
            cells.setdefault((x["model"], x["arm"]), []).append(x)
        md = lambda v: statistics.median(v) if v else None
        for (mod, arm), xs in sorted(cells.items(), key=lambda kv: (ORDER_ARM.get(kv[0][1], 3), kv[0][1], kv[0][0])):
            bars = [x["bar"] for x in xs if x["bar"] is not None and not x["no_bar_reading"]]
            bm = md(bars)
            L.append(f"| {mod} | {arm} | {len(xs)} | {'' if bm is None else f'{bm:.3f}'} | {M(md([x['all']['total'] for x in xs]))} | "
                     f"{M(md([x['all']['uncached'] for x in xs]))} | {M(md([x['tokens_per_h'] for x in xs if x['tokens_per_h']]))} | "
                     f"{M(md([x['uncached_per_h'] for x in xs if x['uncached_per_h']]))} | "
                     f"{'' if not xs else round(md([x['requests_per_h'] for x in xs if x['requests_per_h']]) or 0, 1)} | {K(md([x['main_peak_context'] for x in xs]))} |")
        L.append("")
    flagged = [x for x in out if x.get("ledger_copy_rows")]
    notag = [x["run_id"] for x in out if not x.get("all")]
    if flagged or notag:
        L += ["## Checks", ""]
        L += [f"- {x['run_id']}: {x['ledger_copy_rows']}" for x in flagged]
        L += [f"- no cost tag or no ledger rows: {', '.join(notag)}"] if notag else []
        L.append("")
    open(os.path.join(a.out_dir, "harness-tokens.md"), "w").write("\n".join(L))
    print(f"{len(out)} runs ({sum(1 for x in out if x.get('all'))} with ledger rows) -> {a.out_dir}/harness-tokens.md, .json, .csv")

if __name__ == "__main__":
    main()
