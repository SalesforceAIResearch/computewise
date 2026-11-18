"""In-process patch bind (next/ r8.8; analysis ruling (A), 2026-11-02 17:42 MST).

A record is bound to its evaluator's bytes (entry_hash, exact_module, seam_sha256,
module_shas), but a runner can import the bound file and replace its functions in
memory before calling main(): G-kvreph-opus-A3 job 44 replaced
AdaptiveKVQuantizer.quantize_key/quantize_value with an identity and was served the
4-bit record (runner's helper-edit scan, 2026-11-02). This module checks, in the
job's own interpreter, that every function and method the bound file defines is
still the one its bytes compile to.

The checked set comes from the file itself: its source is compiled and every
function and method defined at module level or in a (nested) class body is
expected, so no list a record carries can leave one out. For each expected name
the live object where it is defined (the module attribute, the class attribute
through the MRO) must be one of:
  - the original: its code hashes to the compiled code (co_code, co_consts
    recursively, co_names), so an in-place __code__ swap is caught;
  - a wrapper whose code can call the original: a closure cell or a global its
    own code names holds it, or it is a functools.partial / bound method of it
    (the r4 runner shapes: print/log wrappers around score_prediction, a slicing
    load_workload; analysis ruled these legitimate). __wrapped__ alone does not
    count: functools.wraps sets it on a function that may never call it;
  - one of gpu-mock's own seams (the caller passes them).
Anything else is a mismatch: a lambda (job 44), a nullcontext stand-in
(G-kvreph-haiku-V3 job 11), a function that never refers to the original, a
class put where a function was.

Disclosed limit: a wrapper that holds the original but changes
its arguments or result is accepted. runner's transcript scan flags that shape
(`served_despite_semantic_wrapper`).

Torch-free; the replay layer imports it lazily."""
from __future__ import annotations

import functools
import hashlib
import os
import types

_CO_OPTIMIZED = 0x0001          # set on function bodies, not on module or class bodies

_compiled_cache: dict = {}


def _const_repr(k) -> str:
    """A constant's repr that does not depend on the process: a set or frozenset
    constant (`x in {"a", "b"}` compiles to one) iterates in string-hash order, which
    PYTHONHASHSEED randomises per process, so its plain repr differs between the
    job's import and this check's compile of the same bytes (next/ r8.9.3: harness.py
    patch_qwen's `{"Qwen2Attention", ...}` false-refused the late-pole sparse cell,
    gpu-mock's proof 2026-11-03 09:24Z). Elements are sorted by their own stable
    repr; tuples recurse."""
    if isinstance(k, (frozenset, set)):
        return type(k).__name__ + "{" + ",".join(sorted(_const_repr(x) for x in k)) + "}"
    if isinstance(k, tuple):
        return "(" + ",".join(_const_repr(x) for x in k) + ")"
    return repr(k)


def _code_hash(co) -> str:
    h = hashlib.sha256()

    def feed(c):
        h.update(c.co_code)
        h.update(repr(c.co_names).encode())
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                feed(k)
            else:
                h.update(repr(type(k)).encode())
                h.update(_const_repr(k).encode())
    feed(co)
    return h.hexdigest()


def expected(path: str) -> dict | None:
    """{qualname: set(code hashes)} for every function and method `path` defines
    (module level and class bodies, nested classes included; lambdas,
    comprehensions and functions nested in functions are not names a runner can
    reach). None when the file cannot be read or compiled."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _compiled_cache:
        return _compiled_cache[key]
    try:
        with open(path, "rb") as fh:
            src = fh.read()
        mod = compile(src, path, "exec", dont_inherit=True)
    except Exception:
        _compiled_cache[key] = None
        return None
    out: dict = {}

    def walk(co, prefix: str):
        for k in co.co_consts:
            if not isinstance(k, types.CodeType) or k.co_name.startswith("<"):
                continue
            q = prefix + k.co_name
            if k.co_name.startswith("__annotate"):
                continue                  # compiler-generated (3.14 lazy annotations), never a runner's target
            if k.co_flags & _CO_OPTIMIZED:
                out.setdefault(q, set()).add(_code_hash(k))
            else:
                walk(k, q + ".")          # a class body: its methods
    walk(mod, "")
    _compiled_cache[key] = out
    return out


def _lookup(g: dict, qual: str):
    """The live object at qualname `qual` in the namespace `g`, found where it is
    defined: the module attribute, else the class attribute through the MRO.
    Returns (found, object)."""
    parts = qual.split(".")
    obj = g.get(parts[0], _MISSING)
    if obj is _MISSING:
        return False, None
    for name in parts[1:]:
        if not isinstance(obj, type):
            return False, None
        for base in getattr(obj, "__mro__", (obj,)):
            if name in base.__dict__:
                obj = base.__dict__[name]
                break
        else:
            return False, None
    return True, obj


_MISSING = object()


def _functions_of(obj) -> list:
    """The plain functions a class attribute or module attribute stands for."""
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        return [f for f in (obj.fget, obj.fset, obj.fdel) if f is not None]
    if isinstance(obj, functools.cached_property):
        return [obj.func]
    return [obj]


def _names_in(co) -> set:
    out = set(co.co_names)
    for k in co.co_consts:
        if isinstance(k, types.CodeType):
            out |= _names_in(k)
    return out


def _is_original(f, want: set, own_ids: set, g: dict | None = None) -> bool:
    """The function the file defines: its code hashes to the compiled code and it
    runs in the module's own namespace (a lambda with the same bytecode compiled
    in a runner resolves its names in the runner's globals, not the module's)."""
    if id(f) in own_ids:
        return True                 # an original whose code gpu-mock redirected to its seam
    co = getattr(f, "__code__", None)
    if not isinstance(co, types.CodeType) or _code_hash(co) not in want:
        return False
    return g is None or getattr(f, "__globals__", None) is g


def _holds_original(f, want: set, own_ids: set, g: dict | None = None, depth: int = 0) -> bool:
    """Whether `f` is a wrapper whose code can call the original: a
    functools.partial of it, a bound method of it, or a function that has it in a
    closure cell or in a global its own code names. Two levels deep (a wrapper of
    a wrapper). `__wrapped__` alone is not enough: functools.wraps sets it on a
    function that may never call what it names."""
    if depth > 2 or f is None:
        return False
    if isinstance(f, functools.partial):
        inner = f.func
    elif isinstance(f, types.FunctionType):
        inner = None              # a Python function's __wrapped__ is only a label (functools.wraps)
    else:
        # a bound method, or a C-level decorator object the file's own decorators
        # make (functools.lru_cache / cache): its __wrapped__ is what it calls
        inner = getattr(f, "__func__", None) or getattr(f, "__wrapped__", None)
    if inner is not None and inner is not f:
        if _is_original(inner, want, own_ids, g) or _holds_original(inner, want, own_ids, g, depth + 1):
            return True
    if isinstance(f, types.FunctionType):
        for cell in f.__closure__ or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if callable(v) and (_is_original(v, want, own_ids, g) or _holds_original(v, want, own_ids, g, depth + 1)):
                return True
        fg = f.__globals__
        for name in _names_in(f.__code__):
            v = fg.get(name)
            if v is not None and v is not f and callable(v) and not isinstance(v, type) \
                    and (_is_original(v, want, own_ids, g) or _holds_original(v, want, own_ids, g, depth + 1)):
                return True
    return False


def mismatch(g: dict, path: str, own_ids: set = frozenset(), own_objs: set = frozenset()) -> str | None:
    """The first function or method of the file at `path` whose live object in the
    namespace `g` is neither the original nor a wrapper holding it (see the module
    docstring), as a one-line detail; None when every one is. `own_ids`: ids of
    original functions gpu-mock redirected in place to its seams (still the
    originals); `own_objs`: ids of gpu-mock's own seam replacements (skipped).
    An expected name that is no longer there at all is skipped (calling it would
    fail on hardware too). A file that cannot be compiled is not checked."""
    exp = expected(path)
    if not exp:
        return None
    for qual in sorted(exp):
        found, obj = _lookup(g, qual)
        if not found or id(obj) in own_objs:
            continue
        want = exp[qual]
        if isinstance(obj, type):
            return f"{qual} in {os.path.basename(path)} is now the class {getattr(obj, '__qualname__', obj)!r}, not the function the file defines"
        for f in _functions_of(obj):
            if id(f) in own_objs or _is_original(f, want, own_ids, g):
                continue
            if _holds_original(f, want, own_ids, g):
                continue
            what = getattr(f, "__qualname__", None) or type(f).__name__
            where = getattr(getattr(f, "__code__", None), "co_filename", None)
            return (f"{qual} in {os.path.basename(path)} was replaced in-process by {what!s}"
                    + (f" ({where})" if where else "") + ", which does not call the original")
    return None
