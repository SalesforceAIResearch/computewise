"""Write the build stamp into SKILL.md: a hash of every file the harness runs.

The box's `skill-synced:` check compares SKILL.md alone, so a change to the Python code must change SKILL.md too, or a
machine could run new code under an old key (or old code under a new one). The stamp is the sha256 of the package's
sources, in a sorted, stable order; `--check` reports whether SKILL.md carries the current one.

usage: python3 -I stamp.py [--check]
"""

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["bin/gpuh"] + sorted(os.path.join("gpuharness", f) for f in os.listdir(os.path.join(HERE, "gpuharness"))
                                if f.endswith(".py"))
MARK = re.compile(r"<!-- build: [0-9a-f]{16} -->\n?")


def tree_hash() -> str:
    h = hashlib.sha256()
    for rel in SOURCES:
        with open(os.path.join(HERE, rel), "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read() + b"\0")
    return h.hexdigest()[:16]


def main() -> int:
    path = os.path.join(HERE, "SKILL.md")
    text = open(path).read()
    want = f"<!-- build: {tree_hash()} -->\n"
    have = MARK.search(text)
    if "--check" in sys.argv:
        ok = bool(have) and have.group(0).strip() == want.strip()
        print(("current: " if ok else "stale: SKILL.md carries ") + (have.group(0).strip() if have else "no stamp")
              + ("" if ok else f"; the code is {want.strip()}"))
        return 0 if ok else 1
    new = MARK.sub(want, text) if have else text.rstrip("\n") + "\n\n" + want
    if new != text:
        open(path, "w").write(new)
    print(want.strip(), "->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
