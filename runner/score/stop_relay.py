"""Carry a budget stop from the controller to the VM, so run_one.py records killed-on-budget.

usage, on the controller:  python3 -I stop_relay.py <tag> [--every 15] [--vm-stop-dir /srv/stage/gate/stop]

Why it exists: the budget is enforced on the controller. The the router session's
budget_watch.py reads the ledger there and creates ~/.p006-gate/stop/<tag>, and the controller gate
then answers that run with 429. But run_one.py runs on the VM and decides killed-on-budget by one
rule: whether <its --stop-dir>/<tag> exists on the VM at exit. Nothing joins the two. This polls
the controller file and, once it appears, creates the VM file through gcp's vm.sh. It then asks
run_one.py to stop the run, because a harness refused by the gate may keep retrying rather than
exit. It exits when the run directory on the VM shows an end time.

It reads only the controller's stop directory and writes only the one VM file; it never touches the
ledger, the gate or the model router. Standard library only.

It also exits when its run's claim on the controller has left running or been collected (ledger.run_ended, the token watchers' rule),
read every --every s without an ssh, and it refuses to start without P006_VM (gcp, 2026-11-04 17:22Z: twelve relays restarted
by hand on 11-02 without P006_VM asked the E2 for a status.json that lives on the run's machine, never saw their runs end, and
ran 1 day 20 h, a fresh gcloud every 120 s, on a controller out of memory).
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ledger import run_ended
VM = os.path.normpath(os.path.join(HERE, "..", "..", "gcp", "host", "vm.sh"))
RUN_ONE = "/opt/computewise/isolation/run_one.py"

def vm(cmd):
    return subprocess.run(["bash", VM, "ssh", cmd], capture_output=True, text=True, timeout=120)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--every", type=float, default=15.0)
    # the VM's end check is an ssh (a fresh gcloud each time): every 120 s, not every 15 s, so twelve relays do not start 48 gcloud
    # processes a minute. the controller stop file is still read every --every s, and a
    # stop found there is relayed at once; only the relay's own exit after the run ends waits up to this long
    # 600 s since the claim check: the claim ends the relay within --every s of the driver's
    # collect, so the ssh is only the fallback for a claim the driver never updates
    ap.add_argument("--done-every", type=float, default=600.0)
    ap.add_argument("--controller-stop-dir", default=os.path.expanduser(os.environ.get("P006_GATE_HOME", "~/.p006-gate") + "/stop"))
    ap.add_argument("--vm-stop-dir", default="/srv/stage/gate/stop")
    ap.add_argument("--runs", default="/srv/runs")
    a = ap.parse_args()
    if not os.environ.get("P006_VM"):
        sys.exit("stop_relay: P006_VM is unset, so vm.sh would ask the E2, not the run's machine; start it with the run's "
                 "P006_VM (and P006_ZONE)")
    local = os.path.join(a.laptop_stop_dir, a.tag)
    run_dir = f"{a.runs}/{a.tag}"
    relayed = False
    # the first ssh one --done-every in (2026-11-04: eighteen relays restarted together would start eighteen gcloud at once); a run
    # that has already ended is caught sooner by its claim
    last_done_check = time.time()
    while True:
        if not relayed and os.path.exists(local):
            r = vm(f"sudo mkdir -p {a.vm_stop_dir} && sudo touch {a.vm_stop_dir}/{a.tag} && "
                   f"sudo python3 -I {RUN_ONE} --stop {run_dir}")
            print(f"{time.strftime('%H:%M:%S')} budget stop relayed for {a.tag}: exit {r.returncode}", flush=True)
            relayed = r.returncode == 0
        if run_ended(a.tag):
            print(f"{time.strftime('%H:%M:%S')} run {a.tag}: its claim has ended; relay exits", flush=True)
            return
        if time.time() - last_done_check < a.done_every:
            time.sleep(a.every)
            continue
        last_done_check = time.time()
        done = vm(f"python3 -I -c \"import json,sys; s=json.load(open('{run_dir}/status.json')); "
                  f"sys.exit(0 if s.get('end_utc') else 1)\" 2>/dev/null")
        if done.returncode == 0:
            print(f"{time.strftime('%H:%M:%S')} run {a.tag} has ended; relay exits", flush=True)
            return
        time.sleep(a.every)

if __name__ == "__main__":
    main()
