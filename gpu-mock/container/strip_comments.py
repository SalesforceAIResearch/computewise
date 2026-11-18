#!/usr/bin/env python3
"""Write a copy of a Python source file with its comments and docstrings removed.

Used by install_into_image.sh for the files it installs into the box (the torch
layer and sitecustomize.py), so the box carries code without our prose. The
repository and the release archive keep the commented source. Everything that
is not a comment or a docstring is the original text, character for character,
and the copy is checked to be the same program (the same syntax tree once
docstrings are dropped) before it is written.

    python3 -I strip_comments.py <in.py> <out.py>
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize


def _docstrings(tree) -> dict:
    """(line, col) of each docstring -> True when it is a class or function
    body's only statement (it then becomes `pass`)."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                out[(b[0].lineno, b[0].col_offset)] = len(b) == 1 and not isinstance(node, ast.Module)
    return out


def strip(src: str) -> str:
    """Comments go; a docstring becomes `pass` where it was a block's only
    statement and is dropped otherwise; lines left empty go, and blank runs
    collapse to one."""
    docs = _docstrings(ast.parse(src))
    spans = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            spans.append((tok.start, tok.end, ""))
        elif tok.type == tokenize.STRING and tok.start in docs:
            spans.append((tok.start, tok.end, "pass" if docs[tok.start] else ""))
    lines = src.splitlines(keepends=True)
    touched = set()
    for (sr, sc), (er, ec), rep in sorted(spans, reverse=True):
        head = lines[sr - 1][:sc]
        tail = lines[er - 1][ec:]
        lines[sr - 1:er] = [head + rep + tail]
        touched = {r if r < sr else (r - (er - sr) if r > er else sr) for r in touched}
        touched.add(sr)
    out, blank = [], False
    for i, ln in enumerate(lines, 1):
        body = ln.rstrip("\r\n")
        nl = ln[len(body):] or "\n"
        if i in touched:
            body = body.rstrip()
            if not body.strip():
                continue
        if not body.strip():
            if not blank and out:
                out.append("\n")
            blank = True
            continue
        out.append(body + nl)
        blank = False
    return "".join(out).strip("\n") + "\n"


def _no_doc_dump(src: str) -> str:
    """The module's syntax tree with docstrings removed, for the equivalence check."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                # an empty body is a `pass` everywhere but a module, whose body may be empty
                node.body = b[1:] or ([] if isinstance(node, ast.Module) else [ast.Pass()])
    return ast.dump(tree, include_attributes=False)


def main() -> None:
    src_path, out_path = sys.argv[1], sys.argv[2]
    src = open(src_path, encoding="utf-8").read()
    out = strip(src)
    if _no_doc_dump(src) != _no_doc_dump(out):
        sys.exit(f"strip_comments: {src_path}: the stripped copy is not the same program; not written")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)


if __name__ == "__main__":
    main()
