# usage: python3 -I isolation/tests/budget_watch_known_answer.py <budget_watch.py>
# Known answer for budget_watch's stop and resume, on a scratch ledger and stop dir (never the gate's): it lifts only a
# stop it wrote itself and that is not sticky; any other stop file stays and its refusal is logged once.
import importlib.util, json, os, sqlite3, sys, tempfile
spec = importlib.util.spec_from_file_location("bw", sys.argv[1]); bw = importlib.util.module_from_spec(spec); spec.loader.exec_module(bw)
d = tempfile.mkdtemp(); stop = os.path.join(d, "stop"); led = os.path.join(d, "ledger.sqlite")
c = sqlite3.connect(led); c.execute("CREATE TABLE turns (tag TEXT, month TEXT, usd REAL, upstream_usd REAL)")
month = bw.months_back()[0]
def spend(tag, usd):
    c.execute("DELETE FROM turns WHERE tag=?", (tag,)); c.execute("INSERT INTO turns VALUES (?,?,?,NULL)", (tag, month, usd)); c.commit()
logs = []
def check(ceilings):
    bw.check({"stop_dir": stop, "ledger": led, "ceilings": ceilings}, logs.append)
ok = True
def want(label, cond):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {label}")
T = "P006-T-1"
spend(T, 5.0); check({T: 4.0})
body = json.load(open(os.path.join(stop, T)))
want("over the ceiling: a stop file written by budget_watch", body.get("writer") == "budget_watch" and body.get("ceiling") == 4.0)
check({T: 10.0})
want("ceiling raised: budget_watch lifts its own stop", not os.path.exists(os.path.join(stop, T)) and logs[-1]["event"] == "resume")
cases = [("token-cap stop (sticky, watch-tokens)", {"writer": "watch-tokens", "sticky": True, "tokens": 1}),
         ("empty file (older watch-tokens, or a person)", None),
         ("budget_watch's own JSON marked sticky", {"writer": "budget_watch", "run": T, "usd": 5, "ceiling": 4.0, "sticky": True}),
         ("another run's stop body", {"writer": "budget_watch", "run": "P006-other", "usd": 5, "ceiling": 4.0})]
for label, b in cases:
    with open(os.path.join(stop, T), "w") as f:
        if b is not None:
            json.dump(b, f)
    n = len(logs)
    check({T: 10.0}); check({T: 10.0})
    refused = [e for e in logs[n:] if e["event"] == "resume-refused"]
    want(f"{label}: stays, refusal logged once ({len(refused)})", os.path.exists(os.path.join(stop, T)) and len(refused) == 1)
    os.remove(os.path.join(stop, T)); bw.REFUSED.clear()
with open(os.path.join(stop, T), "w") as f:
    json.dump({"run": T, "usd": 5.0, "ceiling": 4.0, "turns": 1, "ts_epoch": 1}, f)
check({T: 10.0})
want("a stop from the previous budget_watch (no writer key): lifted", not os.path.exists(os.path.join(stop, T)))
spend(T, 5.0); check({T: 4.0})
want("no temp file left beside a stop", sorted(os.listdir(stop)) == [T])
sys.exit(0 if ok else 1)
