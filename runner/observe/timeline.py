"""Render one ComputeWise run directory as a self-contained HTML timeline.

usage: python3 -I timeline.py <run_dir> [--out <run_dir>/timeline.html]

One shared time axis, from the agent's start to the end of the scoring window, with:
  - one lane per GPU, busy intervals drawn from gpu-mock's occupancy log (rebuilt from enqueue
    events, as the scorer does), coloured by job;
  - job launches, ends and failures as marks on a jobs lane, with each job's host phase (process
    alive, GPUs idle) drawn hatched;
  - the agent's model turns and tool calls on their own lanes, from the transcript;
  - whole-box-idle intervals shaded across every lane and labelled with their attributed cause;
  - memory used by the box over time, from the cgroup samples, when the run has them;
  - a header with the scorer's numbers, so the picture and the numbers sit together.
Everything comes from the run directory. It reuses the scorer's functions (score/score_run.py and
score/idle.py), so what is drawn is exactly what is scored. No external assets.
"""
import argparse, html, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "score"))
import idle, score_run

CAUSE_COLOUR = {
    "before_first_launch": "#c9a227", "job_host_phase": "#8a8f98", "startup_not_overlapped": "#e07b39", "after_agent_ended": "#b5452e",
    "after_job_failure": "#9b2f8f", "model_turn": "#3a6fb0", "wait_or_poll": "#2f8f6b",
    "non_gpu_tool": "#6b5b95", "unattributed": "#555555", "route_down": "#d62728",
}
JOB_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#9c755f",
               "#bab0ac", "#ff9da7"]

def cause_of(m, first_launch, agent_end, failures, turns, tools, alive, route_down=(), agent_parts=()):
    """The same rule as idle.attribute, for one instant, so the shading matches the totals."""
    if first_launch is not None and m < first_launch:
        return "before_first_launch"
    if any(s <= m < e for s, e in alive):
        return "startup_not_overlapped" if any(s <= m < e for s, e in agent_parts) else "job_host_phase"
    if any(s <= m < e for s, e in route_down):
        return "route_down"
    if agent_end is not None and m > agent_end:
        return "after_agent_ended"
    fail = [f for f in failures if f <= m]
    if fail:
        nxt = min((s for s, e, k, _ in tools if s >= max(fail)), default=None)
        if nxt is None or m < nxt:
            return "after_job_failure"
    acts = sorted(turns + tools)
    return next((k for s, e, k, _ in acts if s <= m < e and k == "model_turn"), None) \
        or next((k for s, e, k, _ in acts if s <= m < e), None) or "unattributed"

def render(run_dir):
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    r = idle.analyse(run_dir, allow_live=True)        # the timeline is how a live run is watched
    live = not idle.finished(run_dir)
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    run = idle.run_inputs(run_dir)
    s0 = next((e for e in ev if e["kind"] == "daemon_start"), ev[0])
    off = s0["t_wall"] - s0["t_mono"]
    gpus = s0.get("gpus") or task.get("box", {}).get("gpus") or 4
    ivs = {g: [(a + off, b + off) for a, b in l] for g, l in score_run.busy_intervals(ev).items()}
    # which daemon job owns each enqueue, so lanes can be coloured by job
    owner = []
    cursor = {}
    for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
        a = max(cursor.get(g, t), t); cursor[g] = a + c
        owner.append((g, a, a + c, e.get("job")))
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    fails = [e for e in ev if e["kind"] == "failure"]
    jobs = task.get("jobs") or []
    tj = {d: (idle.task_job_of(hello[d], lookup.get(d), jobs) if jobs else None) for d in hello}
    labels = sorted({tj[d] or f"job {d}" for d in hello})
    colour = {lab: JOB_PALETTE[i % len(JOB_PALETTE)] for i, lab in enumerate(labels)}
    recs = idle.transcript_records(run_dir)
    # the scorer's rule for a box stopped with the harness running (idle.analyse): the agent ended at teardown
    killed_at = st.get("teardown_wall") if not st.get("agent_end_wall") and not st.get("harness_end_wall") else None
    turns, tools, session_end = idle.activity(recs, open_until=killed_at)
    agent_start = st.get("agent_start_wall") or (idle.ts(recs[0]["timestamp"]) if recs else min(idle.job_start(e) for e in hello.values()))
    agent_end = (st.get("agent_end_wall") or st.get("harness_end_wall") or killed_at) or session_end
    w0 = agent_start
    w1 = w0 + r["window"]["wall_s"]
    first_launch = min((idle.job_start(e) for e in hello.values()), default=None)
    whole, _ = idle.whole_and_partial(ivs, gpus, w0, w1)
    # the scorer's inputs exactly, so the shading matches the totals: launcher time counts as alive,
    # and host phase an overlapping schedule would have hidden is the agent's (idle.startup_split)
    alive = idle.alive_intervals(ev) + idle.launcher_intervals(run_dir)
    rdown = idle.route_down_intervals(run_dir)
    agent_parts, _, _ = idle.startup_split(ev, task, whole, agent_start, run_dir)
    failures = [e["t_wall"] for e in fails]

    W, LEFT, LANE = 1180, 150, 22
    span = max(w1 - w0, 1e-6)
    X = lambda t: LEFT + (min(max(t, w0), w1) - w0) / span * (W - LEFT - 20)
    job_rows = labels
    lanes = [f"GPU {g}" for g in range(gpus)] + [f"job {lab}" for lab in job_rows] + ["model turns", "tool calls"]
    mem_path = os.path.join(run_dir, "resources.jsonl")
    mem = []
    mem_cap = None
    if os.path.exists(mem_path):
        # isolation's cgroup sampler (2026-10-20): a "limits" line, then "sample" lines with t_wall,
        # memory_current, memory_peak, cpu_*_usec, pressure, io, disk_used_bytes, ...
        for line in open(mem_path):
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("kind") == "limits":
                mem_cap = m.get("memory_max") if isinstance(m.get("memory_max"), (int, float)) else None
                continue
            if m.get("kind") not in (None, "sample") or m.get("t_wall") is None:
                continue
            mem.append((float(m["t_wall"]), float(m.get("memory_current") or 0)))
    if mem:
        lanes.append("memory")
    H = 60 + LANE * len(lanes) + (60 if mem else 0) + 40
    y = {name: 50 + i * LANE for i, name in enumerate(lanes)}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="system-ui,sans-serif" font-size="11">',
           '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           '<line x1="0" y1="0" x2="0" y2="6" stroke="#8a8f98" stroke-width="2"/></pattern></defs>']
    # whole-box-idle shading, split by cause; pieces narrower than a pixel are counted, not drawn
    tiny = []
    for a, b in whole:
        cuts = sorted({a, b, *[p for p in (first_launch, agent_end, *failures) if p and a < p < b],
                       *[p for s, e, *_ in turns + tools + alive + rdown + agent_parts for p in (s, e) if a < p < b]})
        pieces = []
        for x0, x1 in zip(cuts, cuts[1:]):
            c = cause_of((x0 + x1) / 2, first_launch, agent_end, failures, turns, tools, alive, rdown, agent_parts)
            if pieces and pieces[-1][2] == c:
                pieces[-1][1] = x1
            else:
                pieces.append([x0, x1, c])
        for x0, x1, c in pieces:
            if X(x1) - X(x0) < 1.0:
                tiny.append(x1 - x0)
                continue
            svg.append(f'<rect x="{X(x0):.1f}" y="44" width="{max(0.5, X(x1) - X(x0)):.1f}" height="{LANE * (gpus + len(job_rows) + 2) + 6}" '
                       f'fill="{CAUSE_COLOUR[c]}" opacity="0.16"><title>whole-box idle: {c}, {x1 - x0:.1f} s</title></rect>')
    for name in lanes:
        svg.append(f'<text x="8" y="{y[name] + 14}" fill="#333">{html.escape(name)}</text>')
        svg.append(f'<line x1="{LEFT}" y1="{y[name] + LANE - 2}" x2="{W - 20}" y2="{y[name] + LANE - 2}" stroke="#eee"/>')
    px = span / (W - LEFT - 20)                 # seconds per pixel
    merged = {}
    for g, a, b, j in sorted(owner, key=lambda o: (o[0], o[3] or 0, o[1])):
        lst = merged.setdefault((g, j), [])
        if lst and a - lst[-1][1] < px:
            lst[-1][1] = max(lst[-1][1], b)
        else:
            lst.append([a, b])
    for (g, j), lst in merged.items():
        lab = tj.get(j) or f"job {j}"
        for a, b in lst:
            if b <= w0 or a >= w1:
                continue
            svg.append(f'<rect x="{X(a):.1f}" y="{y[f"GPU {g}"] + 3}" width="{max(0.5, X(b) - X(a)):.2f}" height="{LANE - 8}" fill="{colour[lab]}">'
                       f'<title>{html.escape(lab)} on GPU {g}: {b - a:.1f} s busy</title></rect>')
    for d, h in hello.items():
        lab = tj.get(d) or f"job {d}"
        e = bye.get(d, {}).get("t_wall", w1)
        first_busy = min((a for g, a, b, j in owner if j == d), default=e)
        yy = y[f"job {lab}"]
        svg.append(f'<rect x="{X(idle.job_start(h)):.1f}" y="{yy + 4}" width="{max(0.5, X(first_busy) - X(idle.job_start(h))):.1f}" height="{LANE - 10}" fill="url(#hatch)">'
                   f'<title>{html.escape(lab)} host phase {first_busy - idle.job_start(h):.1f} s</title></rect>')
        svg.append(f'<line x1="{X(idle.job_start(h)):.1f}" y1="{yy + 2}" x2="{X(idle.job_start(h)):.1f}" y2="{yy + LANE - 4}" stroke="{colour[lab]}" stroke-width="2">'
                   f'<title>launch {html.escape(lab)} ({html.escape(str(lookup.get(d, {}).get("mode") or "no lookup"))}{", " + html.escape(str(lookup.get(d, {}).get("record_id"))) if lookup.get(d, {}).get("record_id") else ""})</title></line>')
        if d in bye:
            svg.append(f'<line x1="{X(e):.1f}" y1="{yy + 2}" x2="{X(e):.1f}" y2="{yy + LANE - 4}" stroke="#222" stroke-width="1">'
                       f'<title>end {html.escape(lab)}</title></line>')
    for f in fails:
        flab = tj.get(f.get("job")) or f"job {f.get('job')}"
        svg.append(f'<text x="{X(f["t_wall"]) - 4:.1f}" y="{y.get(f"job {flab}", y["model turns"]) + 15}" fill="#9b2f8f" font-weight="bold">✕<title>failure: {html.escape(str(f.get("mode")))}</title></text>')
    seen_turns = set()
    for s, e, k, mid in turns:
        if (round(s, 3), round(e, 3)) in seen_turns:
            continue                       # one API response written as several records
        seen_turns.add((round(s, 3), round(e, 3)))
        svg.append(f'<rect x="{X(s):.1f}" y="{y["model turns"] + 4}" width="{max(0.5, X(e) - X(s)):.1f}" height="{LANE - 10}" fill="{CAUSE_COLOUR["model_turn"]}" opacity="0.8">'
                   f'<title>model turn {e - s:.1f} s</title></rect>')
    for s, e, k, det in tools:
        svg.append(f'<rect x="{X(s):.1f}" y="{y["tool calls"] + 4}" width="{max(0.8, X(e) - X(s)):.1f}" height="{LANE - 10}" fill="{CAUSE_COLOUR[k]}" opacity="0.8">'
                   f'<title>{html.escape(det)} ({e - s:.1f} s)</title></rect>')
    if agent_end:
        svg.append(f'<line x1="{X(agent_end):.1f}" y1="40" x2="{X(agent_end):.1f}" y2="{y["tool calls"] + LANE}" stroke="#b5452e" stroke-dasharray="4 3"/>'
                   f'<text x="{X(agent_end) + (3 if X(agent_end) < W - 100 else -3):.1f}" y="38" fill="#b5452e" text-anchor="{"start" if X(agent_end) < W - 100 else "end"}">agent ended</text>')
    if mem:
        peak = max(v for _, v in mem) or 1
        top = max(peak, mem_cap or 0) or 1      # scale to the cap when known, so headroom is visible
        base = y["memory"] + LANE + 50
        pts = " ".join(f"{X(t):.1f},{base - 50 * v / top:.1f}" for t, v in mem if w0 <= t <= w1)
        svg.append(f'<polyline points="{pts}" fill="none" stroke="#4e79a7" stroke-width="1.5"/>'
                   f'<text x="{LEFT}" y="{base + 12}" fill="#666">box memory, peak {peak / 2**30:.2f} GiB'
                   + (f' of a {mem_cap / 2**30:.0f} GiB cap' if mem_cap else '') + '</text>')
        if mem_cap:
            svg.append(f'<line x1="{LEFT}" y1="{base - 50:.1f}" x2="{W - 20}" y2="{base - 50:.1f}" stroke="#b5452e" stroke-dasharray="2 3"/>')
    for i in range(6):
        t = w0 + span * i / 5
        svg.append(f'<text x="{X(t):.1f}" y="{H - 12}" fill="#666" text-anchor="middle">{(t - w0):.0f} s</text>')
    svg.append("</svg>")

    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    causes = "".join(f'<li><span class="sw" style="background:{CAUSE_COLOUR.get(k, "#555")}"></span>{html.escape(k)}: {v:.1f} s</li>'
                     for k, v in r["whole_box_idle_by_cause_recorded_s"].items())
    wr = r["work_remaining_at_window_end"]
    tfl = r["time_to_first_launch_recorded_s"]
    first_txt = "never" if tfl is None else f"{tfl:.0f} s"
    head = (("<p style='background:#f8d7da;padding:6px 10px;font-size:13px'><b>Smoke test.</b> "
             + html.escape(idle.SMOKE_NOTE.format(speed=r["window"]["speed"])) + "</p>") if r.get("smoke_test") else "") + \
           ("<p style='background:#fff3cd;padding:6px 10px;font-size:13px'><b>Live run.</b> It has not finished; "
            "the numbers are provisional and the window is still open.</p>" if live else "") + (f"<h1>{html.escape(str(task.get('task_id') or run.get('task')))} &middot; {html.escape(str(run.get('harness')))}"
            f"{' / ' + html.escape(str(run.get('model'))) if run.get('model') else ''}"
            f"{' / ' + html.escape(str(run.get('policy'))) if run.get('policy') else ''}</h1>"
            f"<p class='meta'>{html.escape(os.path.basename(os.path.abspath(run_dir)))} &middot; status {html.escape(str(st.get('status')))} &middot; "
            f"window {r['window']['wall_s']:.1f} s wall to {html.escape(r['window']['to'])} &middot; speed {r['window']['speed']}</p>"
            f"<div class='nums'><div><b>{pct(r['whole_box_idle_share_of_wall'])}</b>whole-box idle</div>"
            f"<div><b>{pct(r['partial_idle_share_of_gpu_seconds'])}</b>partial idle</div>"
            f"<div><b>{first_txt}</b>to first launch</div>"
            f"<div><b>{r['idle_after_agent_stopped_recorded_s']:.0f} s</b>idle after the agent stopped</div>"
            f"<div><b>{wr['remaining'] if wr['remaining'] is not None else '?'}</b>jobs left at window end</div></div>"
            f"<ul class='causes'>{causes}</ul>")
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>ComputeWise run timeline</title><style>"
            "body{font-family:system-ui,sans-serif;margin:24px;color:#222}h1{font-size:18px;margin:0 0 4px}"
            ".meta{color:#666;font-size:12px;margin:0 0 12px}.nums{display:flex;gap:28px;margin:8px 0 10px}"
            ".nums div{font-size:12px;color:#555}.nums b{display:block;font-size:20px;color:#222}"
            ".causes{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:16px;font-size:12px}"
            ".sw{display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:middle}"
            "</style></head><body>" + head + "\n".join(svg)
            + (f"<p class='meta'>{len(tiny)} whole-box-idle intervals narrower than one pixel ({sum(tiny):.2f} s in total) are counted in the numbers above but not drawn.</p>" if tiny else "") +
            "<p class='meta'>Shaded bands are whole-box-idle time, coloured by attributed cause. Hatched marks on a job's row are its host phase (process alive, no GPU busy). "
            "Hover any mark for details. Built by experiments/runner/observe/timeline.py from this run directory only.</p></body></html>")
    return page

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "timeline.html")
    open(out, "w").write(render(a.run_dir))
    print(out)

if __name__ == "__main__":
    main()
