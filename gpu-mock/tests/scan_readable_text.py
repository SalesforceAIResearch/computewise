#!/usr/bin/env python3
"""What an agent can READ in the box that names the mock, by file kind (run
inside a box: `python3 -I scan_readable_text.py [roots...]`). For every text
file under the roots (default: /usr/local/nvidia, site-packages' sitecustomize
and .pth files, /etc), the count of mock words and, for each file kind, where
they sit: in prose (comments, docstrings, YAML comments) or in code text
(identifiers, string literals). Prints one line per kind, then the 40 files
with the most prose hits, with their first prose hit."""
import ast, io, os, re, site, sys, tokenize

WORDS = re.compile(r"gpu[-_ ]?mock|\bmock|fake|simulat|emulat|ComputeWise|printing.press", re.I)
roots = sys.argv[1:] or ["/usr/local/nvidia", "/etc"]
SITE = site.getsitepackages()[0]
files = [os.path.join(SITE, "sitecustomize.py")] + [os.path.join(SITE, x) for x in os.listdir(SITE) if x.endswith(".pth")]
for r in roots:
    for d, _, ns in os.walk(r):
        if "__pycache__" in d:
            continue
        files += [os.path.join(d, n) for n in ns]


def text(p):
    try:
        with open(p, "rb") as fh:
            b = fh.read(4 * 1024 * 1024)
        if b"\0" in b[:8192]:
            return None
        return b.decode("utf-8", "replace")
    except OSError:
        return None


def py_split(src):
    """(prose, code) hit counts of a Python file: comments and docstrings
    are prose; everything else is code."""
    prose = code = 0
    try:
        docs = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                b = node.body[0]
                if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) and isinstance(b.value.value, str):
                    docs.add(b.lineno)
        first = None
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            n = len(WORDS.findall(t.string))
            if not n:
                continue
            if t.type == tokenize.COMMENT or (t.type == tokenize.STRING and t.start[0] in docs):
                prose += n
                first = first or t.string.strip()[:100]
            else:
                code += n
        return prose, code, first
    except (SyntaxError, tokenize.TokenError, ValueError):
        n = len(WORDS.findall(src))
        return n, 0, None


def other_split(p, src):
    """YAML / shell / config: lines starting with # are prose."""
    prose = code = 0
    first = None
    for ln in src.splitlines():
        n = len(WORDS.findall(ln))
        if not n:
            continue
        if ln.lstrip().startswith("#"):
            prose += n
            first = first or ln.strip()[:100]
        else:
            code += n
            if p.endswith((".yaml", ".yml", ".md", ".txt", ".json", ".sh", ".cfg", ".conf")):
                first = first or ("[value] " + ln.strip()[:92])
    return prose, code, first


kinds = {}
top = []
for p in files:
    src = text(p)
    if src is None:
        continue
    if p.endswith(".py"):
        pr, co, first = py_split(src)
    else:
        pr, co, first = other_split(p, src)
    if not (pr or co):
        continue
    where = "nvcompat" if "/pylib/nvcompat/" in p else ("torch_layer" if "/usr/local/nvidia/python/" in p else
            ("sitecustomize" if p.endswith("sitecustomize.py") else os.path.dirname(p)))
    ext = os.path.splitext(p)[1] or "(none)"
    k = (where if where in ("nvcompat", "torch_layer", "sitecustomize") else "other", ext)
    a = kinds.setdefault(k, [0, 0, 0])
    a[0] += 1; a[1] += pr; a[2] += co
    top.append((pr, co, p, first))
print("kind  ext  files  prose_hits  code_hits")
for (w, e), (f, pr, co) in sorted(kinds.items()):
    print(f"{w}  {e}  {f}  {pr}  {co}")
print("--- files outside nvcompat's .py, every one (prose, code, path, first hit):")
for pr, co, p, first in sorted(top, key=lambda x: (-x[0], -x[1])):
    if "/pylib/nvcompat/" in p and p.endswith(".py"):
        continue
    print(pr, co, p, "|", first)
print("--- nvcompat .py with prose hits:", sum(1 for pr, co, p, f in top if "/pylib/nvcompat/" in p and p.endswith(".py") and pr))
