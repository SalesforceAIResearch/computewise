# usage (on a fleet machine, as root, with a box image): python3 -I isolation/tests/bash_timeout_known_answer.py <run_one.py> <image>
# Known answer for --bash-timeout-ms: parse_args, the run.json record and the box env run_one builds, and that a box
# started with that env (as run_one starts it, one `-e K=V` per entry) reads the values in its own environment.
import importlib.util, subprocess, sys
spec = importlib.util.spec_from_file_location("ro", sys.argv[1]); ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
img = sys.argv[2]
ok = True
for label, extra, want in (("absent", [], (36000000, 36000000)), ("product defaults", ["--bash-timeout-ms", "120000,600000"], (120000, 600000))):
    a = ro.parse_args(["--task", "pkg-p6-par-q", "--out", "/tmp/x", "--cost-tag", "t"] + extra)
    got = ro.BASH_TIMEOUTS_MS(a)
    env = {"BASH_DEFAULT_TIMEOUT_MS": str(got[0]), "BASH_MAX_TIMEOUT_MS": str(got[1])}
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "1g", "--entrypoint", "sh"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    out = subprocess.run(cmd + [img, "-c", "env | grep ^BASH_ | sort"], capture_output=True, text=True).stdout.split()
    box = tuple(int(dict(x.split("=", 1) for x in out)[k]) for k in ("BASH_DEFAULT_TIMEOUT_MS", "BASH_MAX_TIMEOUT_MS"))
    good = got == want and box == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {label}: run_one {got}, in the box {box}, want {want}")
for bad in ("120000", "600000,120000", "x,y"):
    try:
        ro.parse_args(["--task", "t", "--out", "/tmp/x", "--cost-tag", "t", "--bash-timeout-ms", bad]); r = "accepted"
    except SystemExit:
        r = "refused"
    ok &= r == "refused"
    print(f"{'PASS' if r == 'refused' else 'FAIL'} --bash-timeout-ms {bad}: {r}")
try:
    ro.parse_args(["--task", "t", "--out", "/tmp/x", "--cost-tag", "t", "--harness", "codex", "--bash-timeout-ms", "120000,600000"]); r = "accepted"
except SystemExit:
    r = "refused"
ok &= r == "refused"
print(f"{'PASS' if r == 'refused' else 'FAIL'} codex harness with --bash-timeout-ms: {r}")
sys.exit(0 if ok else 1)
