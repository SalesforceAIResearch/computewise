"""Set one run's dollar ceiling for the controller's budget watcher, and start the watcher if none runs.

usage, on the controller:  python3 -I set_ceiling.py <tag> <usd>

The watcher is the the router session's budget_watch.py (experiments/isolation/forwarder/). It re-reads
~/.p006-gate/budget.json on every check, and when a run's spend in the ledger reaches its ceiling it
creates ~/.p006-gate/stop/<tag>, which the controller gate answers with 429. This helper only edits that
config, atomically (a temporary file renamed over it, as the forwarder's AGENTS.md asks, so the
watcher never reads a half-written file), and starts the watcher when it is not already running.
It never touches the ledger, the gate or the model router. Standard library only.
"""
import json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
WATCHER = os.path.normpath(os.path.join(HERE, "..", "..", "isolation", "forwarder", "budget_watch.py"))
G = os.path.expanduser(os.environ.get("P006_GATE_HOME", "~/.p006-gate"))
CFG = os.path.join(G, "budget.json")

def running():
    p = subprocess.run(["pgrep", "-f", f"budget_watch.py {CFG}"], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tag, usd = sys.argv[1], float(sys.argv[2])
    if usd <= 0:
        sys.exit("the ceiling must be a positive dollar amount")
    os.makedirs(os.path.join(G, "stop"), exist_ok=True)
    cfg = {"stop_dir": os.path.join(G, "stop"), "interval_s": 15, "ceilings": {},
           "log_path": os.path.join(G, "budget.jsonl")}
    if os.path.exists(CFG):
        cfg.update(json.load(open(CFG)))
    cfg.setdefault("ceilings", {})[tag] = usd
    fd, tmp = tempfile.mkstemp(dir=G, prefix=".budget.", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh, indent=1)
    os.replace(tmp, CFG)
    print(f"ceiling ${usd:.2f} set for {tag} in {CFG}")
    if not running():
        subprocess.Popen([sys.executable, "-I", WATCHER, CFG], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        time.sleep(1)
    pids = running()
    print(f"budget_watch.py {'running: pid ' + ','.join(map(str, pids)) if pids else 'FAILED TO START'}")
    sys.exit(0 if pids else 1)

if __name__ == "__main__":
    main()
