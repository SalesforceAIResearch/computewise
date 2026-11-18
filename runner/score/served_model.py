"""G4: check that every model call in a run was served by the run's own model (campaign gap G4, the study
2026-10-20). On 2026-10-20 the router served Fable requests with Opus 5.5 and nothing in the run said so.

usage: python3 -I served_model.py <run_dir> [--json]

Reads <run_dir>/ledger.jsonl, the run's ledger turns (score/ledger.py dump, written there by the campaign driver,
gap G2), and the run's model from run.json's inputs. Per turn the ledger records `requested_model` (what Claude
Code asked for) and `model` (what the the router says served it). Rule, agreed with the study:
  void   any turn that returned 200 was served a model other than the run's model, or any turn requested a model
         other than the run's model (every call in a run goes to the arm's model, Claude Code's background calls
         included: campaign.json harness.small_fast_model)
  listed a failed turn (not 200) served by another model: it produced no tokens, so it is reported, not voiding
Model ids are compared after dropping a "[1m]" suffix, a provider prefix ("global.anthropic.", "us.anthropic.") and the LLM
provider's trailing "-vertex" (a Vertex AI deployment of the same model, counted as gateway_vertex_relabel_rows).
A scripted run makes no model calls: it passes when its ledger has no turns.
Routing rules specific to an upstream router are read from $CW_ROUTES (a JSON file; none by default):
  "fallback": {model: [fallback model id, expected backend]}: rows served by the fallback model are flagged and
              counted (route_fallback_rows, route_fallback_share), never accepted silently and never voiding by
              themselves; a row of the model on any other backend is counted as route_off_backend_rows;
  "limit_backend": the backend whose 429 rate-limit rows mark a capped subscription (a run held at the limit is void);
  "failover": {"model_prefix": ..., "to_backend": ...}: rows of that model refused 403 on that backend after a
              failover void the run.
A model run with no ledger.jsonl is "unchecked", never "valid".
"""
import json, os, re, sys

def _routes():
    p = os.environ.get("CW_ROUTES")
    return json.load(open(p)) if p and os.path.exists(p) else {}
ROUTES = _routes()
# a model served on one route, with its known fallback model id and the backend its own rows should carry
ROUTE = {k: tuple(v) for k, v in (ROUTES.get("fallback") or {}).items()}
LIMIT_BACKEND = ROUTES.get("limit_backend")
FAILOVER = ROUTES.get("failover") or {}

def norm(m):
    m = (m or "").strip()
    m = re.sub(r"\[1m\]$", "", m)
    # the router's effort suffix: the router strips it, so
    # the ledger's served model is gpt-6-astra while requested_model keeps it; the run's model is the part before the colon
    m = re.sub(r":(minimal|low|medium|high|xhigh)$", "", m)
    # the router's backend pin (model@backend): a route, not another model
    m = re.sub(r"@[a-z0-9_-]+$", "", m)
    # a Vertex AI deployment of the same model, labelled with a trailing -vertex (claude-opus-5-5-vertex served every Opus 5.5 call
    # late in the study; G-k2lp-opus-A5-r15): a provider, not another model. Only a
    # trailing -vertex on the same base id is dropped, so claude-opus-5-vertex stays claude-opus-5 and never matches claude-opus-5-5
    m = re.sub(r"-vertex$", "", m)
    return re.sub(r"^(global|us|eu|apac)\.anthropic\.", "", m)

def vertex_relabel(m):
    """True when the served id carries a provider's -vertex suffix (counted per run as gateway_vertex_relabel_rows)."""
    return bool(re.search(r"-vertex$", re.sub(r"\[1m\]$", "", (m or "").strip())))

# the first call of every Claude Code session on the router's Responses bridge is refused 400 ("messages: role 'system' is not supported
# by this Responses bridge"), and Claude Code retries without the system role. Excused, and listed
SUB_LIMIT_RE = re.compile(r"exceed your account's rate limit|credentials for model \S+ are cooling down|rate_limit_error", re.I)
SUB_LIMIT_VOID_S = 300.0
BRIDGE_400_RE = re.compile(r"role 'system' is not supported by this Responses bridge")

def check(run_dir):
    rj = json.load(open(os.path.join(run_dir, "run.json"))) if os.path.exists(os.path.join(run_dir, "run.json")) else {}
    inp = rj.get("inputs") or {}
    model = inp.get("model") or rj.get("model")
    harness = inp.get("harness") or rj.get("harness")
    p = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.exists(p):
        if harness == "scripted":
            return {"verdict": "unchecked", "reason": "no ledger.jsonl (a scripted run makes no model calls)",
                    "model": None, "turns": None}
        return {"verdict": "unchecked", "reason": "no ledger.jsonl in the run directory", "model": model, "turns": None}
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if harness == "scripted" or not model:
        ok = not rows
        return {"verdict": "valid" if ok else "void", "model": None, "turns": len(rows),
                "reason": "scripted run, no model turns" if ok else f"scripted run with {len(rows)} ledger turns under its tag"}
    want = norm(model)
    # the harness's own side calls on another model (Grok Build's session-title requests on grok-4.6; the analysis
    # 2026-10-26 20:09Z): the ids grok_updates.py / opencode_db.py classified, which score_all writes as side_calls.json beside
    # the ledger. Those rows only are excused, and listed; every other row is checked as before
    sp = os.path.join(run_dir, "side_calls.json")
    side = {x["ledger_id"]: x for x in (json.load(open(sp)) if os.path.exists(sp) else [])}
    side_excused = [{"id": r.get("id"), "model": r.get("model"), "kind": side[r.get("id")].get("kind")}
                    for r in rows if r.get("id") in side and norm(r.get("model")) != want]
    rows = [r for r in rows if not (r.get("id") in side and norm(r.get("model")) != want)]
    bridge_400 = [r.get("id") for r in rows if r.get("status") == 400 and BRIDGE_400_RE.search(str(r.get("error") or ""))]
    rows = [r for r in rows if r.get("id") not in set(bridge_400)]
    # API errors the harness retried itself (Claude Code retries 429s, 5xx and mid-stream errors; Codex treats a capacity error as
    # final): counted as infrastructure, never agent turns, a harness difference recorded per run
    retried_errors = [{"id": r.get("id"), "status": r.get("status"), "error": str(r.get("error") or "")[:120]} for r in rows
                      if r.get("status") not in (200, None)]
    # a routed model's fallback rows ($CW_ROUTES "fallback") are flagged and counted, never accepted silently and never voiding
    # by themselves, so analysis can rule on a mixed run; every other mismatch voids as before
    fallback = []
    if want in ROUTE:
        fb_model, want_backend = ROUTE[want]
        for r in list(rows):
            if norm(r.get("model")) == fb_model:
                fallback.append({"id": r.get("id"), "served": r.get("model"), "backend": r.get("backend"), "status": r.get("status")})
                rows.remove(r)
        off_backend = [r.get("id") for r in rows if norm(r.get("model")) == want and r.get("backend") not in (None, want_backend)]
    else:
        off_backend = []
    served_bad, requested_bad, failed_listed = [], [], []
    for r in rows:
        got, req, ok200 = norm(r.get("model")), norm(r.get("requested_model")), r.get("status") == 200
        rid = r.get("id")
        if req and req != want:
            requested_bad.append({"id": rid, "requested": r.get("requested_model"), "served": r.get("model"), "status": r.get("status")})
        if got != want:
            (served_bad if ok200 else failed_listed).append({"id": rid, "requested": r.get("requested_model"),
                                                            "served": r.get("model"), "status": r.get("status")})
    # a capped subscription's limit ($CW_ROUTES "limit_backend"): the router answers 429 with a rate-limit message on that
    # backend. A run held at the limit is void (subscription_limit), never scored against the agent; a brief limit the harness
    # retried through is flagged and counted. Held: the limit stalled the run SUB_LIMIT_VOID_S or more in all (each limited row
    # to the next 200 after it), or the run ended still limited (no 200 after its last limited row)
    lim = sorted((float(r.get("ts_epoch") or 0), r) for r in rows if r.get("status") == 429 and LIMIT_BACKEND and r.get("backend") == LIMIT_BACKEND
                 and SUB_LIMIT_RE.search(str(r.get("error") or "")))
    ok_t = sorted(float(r.get("ts_epoch") or 0) for r in rows if r.get("status") == 200)
    spans, ended_limited = [], False
    for t, _r in lim:
        nxt = next((x for x in ok_t if x > t), None)
        if nxt is None:
            ended_limited = True
            continue
        spans.append((t, nxt))
    spans.sort(); stall, cur = 0.0, None
    for a, b in spans:
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur: stall += cur[1] - cur[0]
            cur = (a, b)
    if cur: stall += cur[1] - cur[0]
    sub_void = bool(lim) and (stall >= SUB_LIMIT_VOID_S or ended_limited)
    # a failover route ($CW_ROUTES "failover"): when the router fails a model over to a backend that refuses it 403, the harness
    # does not retry a 403 and the turn ends on an API error. Any such row voids the run under the study's "abandon" rule; a 403
    # at the run's very first call with no routed_from is the pool not yet loaded: infrastructure, listed, not a limit
    fv = [r for r in rows if FAILOVER and str(norm(r.get("model"))).startswith(FAILOVER.get("model_prefix", "\0"))
          and r.get("status") == 403 and r.get("backend") == FAILOVER.get("to_backend")]
    first_id = min((r.get("id") or 0) for r in rows) if rows else None
    pool_cold = [r for r in fv if not r.get("routed_from") and r.get("id") == first_id]
    failover_403 = [r for r in fv if r not in pool_cold]
    if failover_403:
        sub_void = True
    rows = [r for r in rows if r not in fv]
    void = bool(served_bad or requested_bad or sub_void)
    reasons = []
    if served_bad:
        reasons.append(f"{len(served_bad)} successful turn(s) served by another model: "
                       + ", ".join(sorted({x['served'] or '?' for x in served_bad})))
    if sub_void:
        if failover_403:
            reasons.append(f"subscription_limit: {len(failover_403)} row(s) failed over to another backend and were refused 403 "
                           f"(first id {failover_403[0].get('id')}, routed_from {failover_403[0].get('routed_from')}); the study's abandon rule")
        if lim and (stall >= SUB_LIMIT_VOID_S or ended_limited):
            reasons.append(f"subscription_limit: {len(lim)} row(s) refused 429 at the subscription's limit, stalling the run "
                           f"{round(stall)} s" + (" and it ended still limited" if ended_limited else ""))
    if requested_bad:
        reasons.append(f"{len(requested_bad)} turn(s) requested another model: "
                       + ", ".join(sorted({x['requested'] or '?' for x in requested_bad})))
    relabel = [r.get("id") for r in rows if vertex_relabel(r.get("model")) and norm(r.get("model")) == want]
    return {"verdict": "void" if void else "valid", "model": model, "turns": len(rows),
            "gateway_vertex_relabel_rows": len(relabel),
            "gateway_vertex_relabel_labels": sorted({r.get("model") for r in rows if r.get("id") in set(relabel)}),
            "turns_200": sum(r.get("status") == 200 for r in rows),
            "served_models": sorted({r.get("model") for r in rows if r.get("model")}),
            "reason": "; ".join(reasons) if reasons else "every turn requested and served by the run's model",
            "served_mismatch": served_bad, "requested_mismatch": requested_bad, "failed_turns_other_model": failed_listed,
            "side_calls_excused": side_excused,
            "route_fallback_rows": len(fallback), "route_fallback_items": fallback[:20],
            "route_fallback_share": round(len(fallback) / (len(rows) + len(fallback)), 4) if (rows or fallback) else None,
            "route_off_backend_rows": len(off_backend),
            "bridge_system_role_400_excused": len(bridge_400),
            "subscription_limit": sub_void, "subscription_limit_rows": len(lim), "subscription_limit_stall_s": round(stall, 1),
            "subscription_limit_ended_limited": ended_limited,
            "failover_403_rows": len(failover_403), "failover_403_items": [{"id": r.get("id"), "routed_from": r.get("routed_from"),
                                                                                          "ts": r.get("ts"), "latency_ms": r.get("latency_ms"), "error": str(r.get("error") or "")[:120]} for r in failover_403[:10]],
            "pool_cold_403_rows": len(pool_cold),
            "billing": sorted({r.get("billing") for r in rows if r.get("billing")}),
            "api_error_rows": len(retried_errors), "api_error_items": retried_errors[:20],
            "api_error_by_status": {str(k): sum(1 for x in retried_errors if x["status"] == k) for k in sorted({x["status"] for x in retried_errors}, key=str)},
            "route_flag": bool(fallback or off_backend)}

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = check(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
    else:
        print(f"served model    {r['verdict'].upper()}: {r['reason']}")
    sys.exit(0 if r["verdict"] != "void" else 3)

if __name__ == "__main__":
    main()
