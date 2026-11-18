"""What a change to decompose.py moved: two of its --json outputs over the same runs, per run and pooled as its L0
table pools them.

usage: python3 -I compare_decompose.py <before.json> <after.json> [--min 0.02]
Prints every run whose loss moved by more than 60 s or any share by more than --min, then the pooled L0 rows by
package and family (Claude: Haiku, Sonnet, Opus; OpenAI: Astra, Sol) before and after.
"""
import argparse, json

K = ("delay", "ordering", "unneeded", "unruled", "harness_gap")
FAMS = (("Claude", ("haiku", "sonnet", "opus")), ("OpenAI", ("astra", "sol")))


def shares(sp):
    return {k: (sp[k] / sp["loss"] if sp["loss"] > 0 else 0.0) for k in K}


def pooled(runs, pk, rd, arm, ms):
    rs = [r for r in runs if r["package"] == pk and r["arm"] == arm and r["model"] in ms and rd in r["split"]]
    if not rs:
        return None
    loss = sum(r["split"][rd]["loss"] for r in rs)
    return dict(n=len(rs), loss=loss, **{k: sum(r["split"][rd][k] for r in rs) / loss for k in K})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--min", type=float, default=0.02)
    a = ap.parse_args()
    A = {r["run_id"]: r for r in json.load(open(a.before))["runs"]}
    B = {r["run_id"]: r for r in json.load(open(a.after))["runs"]}
    both = sorted(set(A) & set(B))
    print(f"runs: {len(A)} before, {len(B)} after, {len(both)} in both")
    print("\n| run | reading | floor, s | loss, h | " + " | ".join(K) + " |\n" + "|---" * (4 + len(K)) + "|")
    n = 0
    for k in both:
        for rd in A[k]["split"]:
            x, y = A[k]["split"][rd], B[k]["split"][rd]
            sx, sy = shares(x), shares(y)
            if abs(y["loss"] - x["loss"]) <= 60 and max(abs(sy[c] - sx[c]) for c in K) <= a.min:
                continue
            n += 1
            print(f"| {k} | {rd or '-'} | {A[k]['floor'][rd]:,.0f} → {B[k]['floor'][rd]:,.0f} | "
                  f"{x['loss'] / 3600:.2f} → {y['loss'] / 3600:.2f} | "
                  + " | ".join(f"{sx[c]:.2f} → {sy[c]:.2f}" if abs(sy[c] - sx[c]) > a.min else f"{sy[c]:.2f}" for c in K) + " |")
    print(f"\n{n} run-readings moved")
    print("\nPooled L0, shares of the summed loss, before → after:\n\n| package | models | n | loss, h | "
          + " | ".join(K) + " |\n" + "|---" * (4 + len(K)) + "|")
    for pk in ("G1", "H5", "G4", "G4n A", "G4n B"):
        for rd in ("", "b", "a"):
            for fl, ms in FAMS:
                p, q = pooled(list(A.values()), pk, rd, "L0", ms), pooled(list(B.values()), pk, rd, "L0", ms)
                if not p or not q:
                    continue
                print(f"| {pk}{' (' + rd + ')' if rd else ''} | {fl} | {q['n']} | {p['loss'] / 3600:.1f} → {q['loss'] / 3600:.1f} | "
                      + " | ".join(f"{p[c]:.2f} → {q[c]:.2f}" if abs(q[c] - p[c]) > 0.005 else f"{q[c]:.2f}" for c in K) + " |")


if __name__ == "__main__":
    main()
