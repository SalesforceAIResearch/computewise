"""Content digests of the checkpoints an agent's ensemble index names, read on the host before the disk image goes.

Called by run_one.py at teardown for a task with `capture.weights_index` (the SF lane's checkpoint-collision package,
the study 2026-11-02 16:55Z, key fix:capture-weights-digest): which member's weights survived can only be read from
the files, and the box's disk image is deleted at teardown. Nothing here imports torch.

A torch.save checkpoint is a zip: `<prefix>/data.pkl` (the object tree, pickled), `<prefix>/data/<key>` (one record per
tensor storage, raw bytes) and a few metadata records. The prefix is the file's own name at save time, so two saves of
identical weights under different names differ byte for byte. Each file gets two digests (SF lane, 2026-11-02 10:04 MST):
- full_digest: data.pkl plus every storage record in key order, prefix and metadata records left out. It moves when
  anything saved moves, optimizer state and saved config included.
- model_digest: the model's tensors only, by canonical name (leading `_orig_mod.` and `module.` dropped), dtype, shape
  and element bytes in row-major order. A model saved alone, with its optimizer, or compiled gives the same model_digest.
  Where the model's tensors can't be told apart from the rest, the file gets model_error unsupported_or_repacked: an
  interpretation limit, not evidence of a lost member.
data.pkl is read by a restricted unpickler whose find_class imports nothing: torch's tensor-rebuild functions and
OrderedDict map to stand-ins defined here, and every other global maps to an inert placeholder, recorded in
foreign_globals. Storages come back as references and are never loaded through the pickle. So no class or function named
in the file is imported or called, whatever the pickle says. Every limit hit is labelled; nothing raises to the caller.

  python3 -I weights_digest.py <home> <tmp> <index rel path> [max_files max_bytes_per_file max_seconds]   (a manual check)
"""
import hashlib
import io
import json
import os
import pickle
import struct
import sys
import time
import zipfile

INDEX_MAX_BYTES = 1 << 20
PKL_MAX_BYTES = 64 << 20
GATHER_MAX_ELEMENTS = 1 << 22      # a non-contiguous model tensor is gathered element by element up to this many
DEFAULTS = {"max_files": 40, "max_bytes_per_file": 2 << 30, "max_seconds": 300}
LIMITS = {"max_files": (1, 64), "max_bytes_per_file": (1, 64 << 30), "max_seconds": (1, 1800)}
META = {"byteorder", "version", ".data/version", ".data/serialization_id", ".format_version", ".storage_alignment"}
ITEMSIZE = {"Float": 4, "Double": 8, "Half": 2, "BFloat16": 2, "Long": 8, "Int": 4, "Short": 2, "Char": 1, "Byte": 1,
            "Bool": 1, "ComplexFloat": 8, "ComplexDouble": 16, "QInt8": 1, "QUInt8": 1, "QInt32": 4}
DTYPE_ITEMSIZE = {"float32": 4, "float": 4, "float64": 8, "double": 8, "float16": 2, "half": 2, "bfloat16": 2,
                  "int64": 8, "long": 8, "int32": 4, "int": 4, "int16": 2, "short": 2, "int8": 1, "uint8": 1, "bool": 1,
                  "complex64": 8, "complex128": 16, "float8_e4m3fn": 1, "float8_e5m2": 1}
MODEL_KEYS = ("model", "state_dict", "model_state_dict", "model_state", "net", "module", "weights")


class _ODict(dict):
    """collections.OrderedDict's stand-in: a dict that can take BUILD state (a state_dict's _metadata attribute)."""


class _Opaque:
    """Every global outside torch's rebuild set: holds what the pickle gave it, does nothing."""
    def __init__(self, *a, **k):
        self.args = a

    def __setstate__(self, st):
        self.state = st


class _TorchName(_Opaque):
    name = None


class _Ref:
    __slots__ = ("key", "dtype", "itemsize")

    def __init__(self, key, dtype, itemsize):
        self.key, self.dtype, self.itemsize = key, dtype, itemsize


class _Tensor:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, int(offset), tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_rest, **_kw):
    return _Tensor(storage, storage_offset, size, stride)


REBUILD = {("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
           ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_v2,
           ("torch._utils", "_rebuild_parameter"): lambda data, *_a, **_k: data,
           ("torch._utils", "_rebuild_parameter_with_state"): lambda data, *_a, **_k: data,
           ("torch._tensor", "_rebuild_from_type_v2"): lambda func, _t, args, _s=None: func(*args),
           ("collections", "OrderedDict"): _ODict,
           ("torch", "Size"): tuple}


class _Restricted(pickle.Unpickler):
    def __init__(self, f):
        super().__init__(f)
        self.foreign = set()

    def find_class(self, module, name):   # imports nothing, ever
        if (module, name) in REBUILD:
            return REBUILD[(module, name)]
        if module == "torch":                 # storage classes, dtypes, device: named, inert
            return type(f"torch.{name}", (_TorchName,), {"name": name})
        self.foreign.add(f"{module}.{name}")
        return type(f"{module}.{name}", (_Opaque,), {})

    def persistent_load(self, pid):
        # ("storage", <storage class or UntypedStorage>, key, location, numel); since torch 2.x the class may be
        # UntypedStorage with the dtype passed to the rebuild instead, so a missing item size is resolved there
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            nm = getattr(pid[1], "name", None) or ""
            base = nm[: -len("Storage")] if nm.endswith("Storage") else nm
            return _Ref(str(pid[2]), base or None, ITEMSIZE.get(base))
        raise pickle.UnpicklingError("unexpected persistent id")


def limits_of(spec):
    """The task's capture.weights_index with defaults, each bound clamped to its range; (limits, notes)."""
    out, notes = {}, []
    for k, d in DEFAULTS.items():
        v = spec.get(k, d)
        lo, hi = LIMITS[k]
        if not isinstance(v, int) or isinstance(v, bool):
            notes.append(f"{k}={v!r} is not an int; {d} used"); v = d
        if not lo <= v <= hi:
            notes.append(f"{k}={v} outside {lo}..{hi}; clamped"); v = min(max(v, lo), hi)
        out[k] = v
    return out, notes


def shown(real, home, tmp):
    """real as the agent would name it: its in-box home or /tmp path (real is already inside one of them)."""
    for view, root in (("~", home), ("/tmp", tmp)):
        rr = os.path.realpath(root)
        if real == rr or real.startswith(rr + os.sep):
            return view + "/" + os.path.relpath(real, rr)
    return real


def inside(path, roots):
    """The real path when it lies under one of roots (after every symlink is resolved), else None."""
    real = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr + os.sep):
            return real
    return None


def to_host(listed, home, tmp, box_home, cwd="project"):
    """The host path for a path as the agent wrote it, or None when it names nothing in the workspace. The box mounts the
    disk image's home at box_home and its tmp at /tmp, and starts the agent in box_home/<cwd> (run_one.py's PROJECT, its
    docker -w); a relative path is taken from there, as the agent's shell and Python take it (runner, 2026-11-02 19:22Z:
    taken from home, the SF task's project-relative index and members read not_found)."""
    if not isinstance(listed, str) or not listed or "\0" in listed:
        return None
    p = os.path.normpath(os.path.join(cwd, listed)) if not os.path.isabs(listed) else os.path.normpath(listed)
    if not os.path.isabs(p):
        return None if p == ".." or p.startswith("../") else os.path.join(home, p)
    for view, root in ((box_home.rstrip("/"), home), ("/tmp", tmp)):
        if p == view or p.startswith(view + "/"):
            return os.path.join(root, os.path.relpath(p, view))
    return None


def index_paths(home, rel, cwd="project"):
    """(the index's checkpoint paths as written, error label or None). The index is the agent's file: untrusted, capped,
    and only a locator; rel is the task's capture.weights_index.path, taken from the agent's working directory like a
    relative member path. Accepted shapes: {"members": [{"checkpoint": p, ...}]} (the SF lane's), a list of path strings,
    or a list of objects with "checkpoint" or "path"."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.split("/"):
        return [], "index_path_invalid"
    p = os.path.join(home, cwd, rel)
    if inside(p, [home]) is None:
        return [], "index_outside_workspace"
    try:
        if os.path.getsize(p) > INDEX_MAX_BYTES:
            return [], "index_too_large"
        doc = json.load(open(p, "rb"))
    except FileNotFoundError:
        return [], "index_not_found"
    except (OSError, ValueError, RecursionError):
        return [], "index_not_json"
    items = doc.get("members") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        return [], "index_no_member_list"
    out = []
    for it in items:
        v = it if isinstance(it, str) else (it.get("checkpoint") or it.get("path")) if isinstance(it, dict) else None
        out.append(v if isinstance(v, str) else None)
    return out, None


def canonical(name):
    while name.startswith(("_orig_mod.", "module.")):
        name = name.split(".", 1)[1]
    return name


def pick_model(obj):
    """(model mapping, where it came from) or (None, why): the whole object when it is a mapping of tensors, else the
    first of MODEL_KEYS whose value is one. Non-tensor entries (a state_dict's step counters) are left out."""
    def tensors(m):
        if not isinstance(m, dict) or not all(isinstance(k, str) for k in m):
            return None
        t = {k: v for k, v in m.items() if isinstance(v, _Tensor)}
        return t if t and all(isinstance(v, (_Tensor, int, float, bool, str, type(None))) for v in m.values()) else None
    t = tensors(obj)
    if t:
        return t, "top"
    if isinstance(obj, dict):
        for k in MODEL_KEYS:
            t = tensors(obj.get(k))
            if t:
                return t, f"key:{k}"
    return None, "no mapping of tensors at the top or under " + "/".join(MODEL_KEYS)


def _stored_reader(fh, info):
    """read_at(pos, n) on a stored (uncompressed) member, by its absolute offset; None for a compressed one."""
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    fh.seek(info.header_offset)
    hdr = fh.read(30)
    if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
        return None
    n, m = struct.unpack("<HH", hdr[26:30])
    base = info.header_offset + 30 + n + m

    def read_at(pos, k):
        fh.seek(base + pos)
        return fh.read(k)
    return read_at


def model_digest(real, z, prefix, model, deadline):
    """{model_digest, model_tensors, model_bytes} or {model_error, model_detail}."""
    names = {}
    for k, t in model.items():
        c = canonical(k)
        if c in names:
            return {"model_error": "unsupported_or_repacked", "model_detail": f"two tensors canonicalise to {c[:80]}"}
        names[c] = t
    h = hashlib.sha256()
    total = 0
    with open(real, "rb") as fh:
        for c in sorted(names):
            if time.time() > deadline:
                return {"model_error": "time_cap"}
            t = names[c]
            if not isinstance(t.storage, _Ref) or not t.storage.itemsize:
                return {"model_error": "unsupported_or_repacked", "model_detail": f"{c[:80]}: unknown storage type"}
            isz = t.storage.itemsize
            try:
                info = z.getinfo(f"{prefix}data/{t.storage.key}")
            except KeyError:
                return {"model_error": "unsupported_or_repacked", "model_detail": f"{c[:80]}: no storage record"}
            numel = 1
            for s in t.size:
                numel *= s
            contig, acc = [], 1
            for s in reversed(t.size):
                contig.insert(0, acc); acc *= s
            h.update(c.encode() + b"\0" + str(t.storage.dtype).encode() + b"\0" + json.dumps(list(t.size)).encode() + b"\0")
            read_at = _stored_reader(fh, info)
            if numel == 0:
                pass
            elif tuple(contig) == t.stride or numel == 1:
                lo, n = t.offset * isz, numel * isz
                if lo + n > info.file_size:
                    return {"model_error": "unsupported_or_repacked", "model_detail": f"{c[:80]}: view past its storage"}
                if read_at:
                    pos = lo
                    while pos < lo + n:
                        b = read_at(pos, min(8 << 20, lo + n - pos))
                        if not b:
                            return {"model_error": "truncated_or_crc"}
                        h.update(b); pos += len(b)
                else:
                    with z.open(info) as m:
                        m.read(lo); h.update(m.read(n))
            else:
                if numel > GATHER_MAX_ELEMENTS:
                    return {"model_error": "unsupported_or_repacked", "model_detail": f"{c[:80]}: non-contiguous, {numel} elements"}
                span = (t.offset + sum((s - 1) * st for s, st in zip(t.size, t.stride)) + 1) * isz
                if span > info.file_size:
                    return {"model_error": "unsupported_or_repacked", "model_detail": f"{c[:80]}: view past its storage"}
                buf = read_at(0, span) if read_at else z.open(info).read(span)
                idx = [0] * len(t.size)
                out = bytearray()
                for _ in range(numel):
                    e = (t.offset + sum(i * st for i, st in zip(idx, t.stride))) * isz
                    out += buf[e:e + isz]
                    for d in range(len(idx) - 1, -1, -1):
                        idx[d] += 1
                        if idx[d] < t.size[d]:
                            break
                        idx[d] = 0
                h.update(bytes(out))
            total += numel * isz
    return {"model_digest": h.hexdigest(), "model_tensors": len(names), "model_bytes": total}


def digest_file(real, max_bytes, deadline):
    """{full_digest, tensors, storage_bytes, model_* , foreign_globals?} or {error: label, detail}."""
    try:
        size = os.path.getsize(real)
    except OSError as e:
        return {"error": "stat", "detail": str(e)[:120]}
    if size > max_bytes:
        return {"error": "too_large", "detail": f"{size} > {max_bytes}"}
    if not zipfile.is_zipfile(real):
        with open(real, "rb") as fh:
            head = fh.read(4)
        # a zip's local header with no central directory at the end is a zip cut short (a save interrupted or overwritten
        # mid-write), not some other format
        return {"error": "truncated_or_crc" if head == b"PK\x03\x04" else "legacy_format" if head[:1] == b"\x80" else "not_zip"}
    try:
        with zipfile.ZipFile(real) as z:
            names = z.namelist()
            pkl = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
            if len(pkl) != 1:
                return {"error": "no_data_pkl"}
            prefix = pkl[0][: -len("data.pkl")]
            if z.getinfo(pkl[0]).file_size > PKL_MAX_BYTES:
                return {"error": "too_large", "detail": "data.pkl over 64 MB"}
            data = z.read(pkl[0])
            h = hashlib.sha256()
            h.update(b"data.pkl\0"); h.update(len(data).to_bytes(8, "big")); h.update(data)
            keys, total = [], 0
            for n in names:
                rel = n[len(prefix):] if n.startswith(prefix) else n
                if rel.startswith("data/") and not n.endswith("/"):
                    keys.append((rel, n))
                elif rel not in META and rel != "data.pkl" and not n.endswith("/"):
                    return {"error": "unexpected_record", "detail": rel[:120]}
            for rel, n in sorted(keys, key=lambda kv: (len(kv[0]), kv[0])):
                if time.time() > deadline:
                    return {"error": "time_cap"}
                h.update(rel.encode() + b"\0")
                with z.open(n) as fh:      # reads through the CRC: a truncated or damaged record raises here
                    length = 0
                    for chunk in iter(lambda: fh.read(8 << 20), b""):
                        h.update(chunk); length += len(chunk)
                h.update(length.to_bytes(8, "big")); total += length
            rec = {"full_digest": h.hexdigest(), "tensors": len(keys), "storage_bytes": total}
            up = _Restricted(io.BytesIO(data))
            try:
                obj = up.load()
            except Exception as e:     # malformed, or a structure the stand-ins can't take
                rec.update(model_error="pickle_parse", model_detail=f"{type(e).__name__}: {str(e)[:100]}")
                obj = None
            if up.foreign:
                rec["foreign_globals"] = sorted(up.foreign)[:10]
            if obj is not None:
                model, frm = pick_model(obj)
                if model is None:
                    rec.update(model_error="unsupported_or_repacked", model_detail=frm)
                else:
                    rec["model_from"] = frm
                    rec.update(model_digest(real, z, prefix, model, deadline))
            return rec
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError, ValueError, KeyError) as e:
        return {"error": "truncated_or_crc", "detail": f"{type(e).__name__}: {str(e)[:100]}"}


def capture(home, tmp, spec, out_path, box_home="/home/researcher", cwd="project"):
    """Write one JSON line per listed checkpoint to out_path and return the summary. Never raises. home and tmp are the
    host paths of the disk image's two trees; box_home is where the box mounted home (run_one.py's HOME), and cwd the
    agent's working directory relative to it (run_one.py's PROJECT less HOME)."""
    t0 = time.time()
    summary = {"index": None, "listed": 0, "digested": 0, "model_digested": 0, "errors": 0, "notes": []}
    try:
        lim, notes = limits_of(spec if isinstance(spec, dict) else {})
        summary["notes"] += notes
        summary["index"] = spec.get("path") if isinstance(spec, dict) else None
        paths, err = index_paths(home, summary["index"], cwd)
        deadline = t0 + lim["max_seconds"]
        seen = {}
        with open(out_path, "w") as fh:
            if err:
                summary["index_error"] = err
                fh.write(json.dumps({"index": summary["index"], "error": err}) + "\n")
            if len(paths) > lim["max_files"]:
                summary["notes"].append(f"index lists {len(paths)}; first {lim['max_files']} read")
            for i, listed in enumerate(paths[: lim["max_files"]]):
                rec = {"i": i, "listed": listed}
                host = to_host(listed, home, tmp, box_home, cwd) if listed is not None else None
                if listed is None:
                    rec["error"] = "no_path"
                elif host is None:
                    rec["error"] = "outside_workspace"
                else:
                    real = inside(host, [home, tmp])
                    if not os.path.lexists(host):
                        rec["error"] = "not_found"
                    elif real is None:
                        rec["error"] = "outside_workspace"
                    elif real in seen:
                        rec.update(resolved=shown(real, home, tmp), error="duplicate_of", of=seen[real])
                    elif not os.path.isfile(real):
                        rec.update(resolved=shown(real, home, tmp), error="not_a_file")
                    elif time.time() > deadline:
                        rec.update(resolved=shown(real, home, tmp), error="time_cap")
                    else:
                        seen[real] = i
                        st = os.stat(real)
                        rec.update(resolved=shown(real, home, tmp), size=st.st_size, mtime=int(st.st_mtime))
                        rec.update(digest_file(real, lim["max_bytes_per_file"], deadline))
                summary["listed"] += 1
                summary["digested" if "full_digest" in rec else "errors"] += 1
                summary["model_digested"] += "model_digest" in rec
                fh.write(json.dumps(rec) + "\n")
    except Exception as e:   # teardown goes on whatever happens here
        summary["errors"] += 1; summary["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    summary["seconds"] = round(time.time() - t0, 2)
    return summary


if __name__ == "__main__":
    # a manual check: writes weights-digests.jsonl in the current directory and prints the summary
    h, t, rel = sys.argv[1:4]
    spec = {"path": rel}
    for k, v in zip(("max_files", "max_bytes_per_file", "max_seconds"), sys.argv[4:7]):
        spec[k] = int(v)
    print(json.dumps(capture(h, t, spec, os.path.join(os.getcwd(), "weights-digests.jsonl"))))
