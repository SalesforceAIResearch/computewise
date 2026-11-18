"""slurmstepd: runs one batch step as the job's owner. It opens the job's stdout and stderr (creating them as the
owner), changes to the job's working directory, and replaces itself with the batch script, so the script's exit
status is the step's. usage: slurmstepd.py <step.json>"""
import json
import os
import sys


def main():
    try:
        import ctypes
        ctypes.CDLL(None).prctl(15, b"slurmstepd", 0, 0, 0)
    except Exception:
        pass
    st = json.load(open(sys.argv[1]))
    try:                                  # a job is an ordinary process to the OOM killer, as on a compute node:
        with open("/proc/self/oom_score_adj", "w") as f:   # raising the score needs no privilege
            f.write("0")
    except OSError:
        pass
    mode = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if st.get("open_mode") == "append" else os.O_TRUNC)
    try:
        fo = os.open(st["stdout"], mode, 0o644)
        fe = os.open(st["stderr"], mode, 0o644) if st["stderr"] != st["stdout"] else fo
    except OSError as e:
        sys.stderr.write("slurmstepd: error: Could not open stdout file %s: %s\n" % (st["stdout"], e.strerror))
        os._exit(1)
    os.dup2(fo, 1)
    os.dup2(fe, 2)
    try:
        os.chdir(st["workdir"])
    except OSError:
        os.write(2, ("slurmstepd: error: couldn't chdir to `%s': %s: going to /tmp instead\n"
                     % (st["workdir"], "No such file or directory")).encode())
        os.chdir("/tmp")
    try:
        os.execve(st["script"], [st["script"]] + st.get("argv", []), st["env"])
    except OSError as e:
        os.write(2, ("slurmstepd: error: execve(): %s: %s\n" % (st["script"], e.strerror)).encode())
        os._exit(1)


if __name__ == "__main__":
    main()
