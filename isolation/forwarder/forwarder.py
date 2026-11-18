#!/usr/bin/env python3
"""The agent's only route to the model API. One program, two modes.

  agent box --(innocuous hostname)--> EDGE --(tunnel or local bridge)--> GATE --> upstream model router or API
                                      per run, beside the agent box      controller loopback, one per machine

The upstream is whatever serves the model API for the study: a local model router that records each
request's cost against a tag, or the provider's API directly. The gate is the only thing that talks to it,
and it lets only model API calls through. The edge runs beside each agent box. It stamps every
request with its run and a request id, so the two sides' logs line up without synchronised
clocks.

Both modes:
  * pass POST /v1/messages, /v1/messages/count_tokens and /v1/responses, and serve the
    configured model list on GET /v1/models; every other request gets a generic 404;
  * refuse a model outside allowed_models (when set) with the API's own 404 "model: <id>";
  * drop the client's credentials and any header it sends under drop_header_prefixes (the router's
    control headers and X-Fwd-*);
  * pass request bodies unchanged and success responses byte-for-byte. Streams are relayed
    line by line as they arrive. The only rewrite is the message text of an error that matches
    leak_pattern (it names the upstream's internals). No Server header is sent;
  * never queue: a request goes upstream as soon as its body is read and checked, and that
    interval is logged as forwarder_wait_ms;
  * answer 429 while stop_file exists; the gate also checks <stop_dir>/<run tag>, which
    budget_watch.py creates;
  * append one JSON line per request to log_path, refusals included.

The edge sends X-Fwd-Run, X-Fwd-Session and X-Fwd-Request-Id. The gate turns the first two into the
upstream's own headers (upstream_headers: the cost tag, the session id, and optionally a priority, a
maximum queue wait and a backend pin) and strips every X-Fwd-* header before the upstream. A gate with
pin_backend_by_model also sends the backend header for those models, so a run on that model is served by
that backend or fails visibly. A gate without cost_tag refuses any request that has no X-Fwd-Run, so
untagged spend cannot happen.

Run:  python3 -I forwarder.py <config.json>      (standard library only, Python >= 3.9)
"""
import hmac
import http.client
import http.server
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid

DEFAULTS = {
    "mode": None,                   # "gate" (controller, in front of the upstream) or "edge" (beside the agent)
    "listen_host": None,            # gate: 127.0.0.1; edge: 0.0.0.0
    "listen_port": None,            # gate: 8790; edge: 8080
    "upstream": None,               # gate: required, the model router or API; edge: required (the gate via the tunnel)
    "cost_tag": None,               # edge: required, the run's ledger tag. gate: the tag for requests
                                    #   without X-Fwd-Run (local use); unset = refuse those requests
    "session_id": None,             # edge: default a fresh UUID per process. gate: only with cost_tag
    "priority": None,               # gate: optional priority value, sent as upstream_headers["priority"]
    "max_wait_s": None,             # gate: optional seconds the upstream may queue a request (upstream_headers["max_wait"])
    "allowed_models": [],           # model ids this forwarder accepts; empty = any
    "models": [],                   # ids served on GET /v1/models; empty = 404 there too
    "log_path": None,               # default <mode>.jsonl in the working directory
    "stop_file": None,              # while this path exists, every model call gets 429
    "stop_dir": None,               # gate: while <stop_dir>/<run tag> exists, that run gets 429
    "run_tag_re": r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    "link_token_file": None,        # both: a file holding a shared secret. The edge sends it as
                                    #   X-Fwd-Key; a gate that has one refuses requests without it
    "upstream_timeout_s": 1800,     # per socket read
    "scrub_errors": True,
    "pin_backend_by_model": {},     # gate: {model id: upstream backend name}. A matching request carries
                                    #   upstream_headers["backend"]: served by that backend or fails
    "upstream_headers": {           # gate: the header names the upstream reads; a missing or null entry is not sent
        "cost_tag": "X-Cost-Tag", "session": "X-Session-Id", "priority": None, "max_wait": None, "backend": None},
    "drop_header_prefixes": ["x-cost-", "x-fwd-"],   # client headers never forwarded (case-insensitive prefixes)
    "leak_pattern": None,           # regex: error messages matching it are replaced by a generic one; default LEAK
    "strip_error_prefix": None,     # regex removed from the start of error messages (an upstream's own name prefix)
}
MODE_DEFAULTS = {"gate": {"listen_host": "127.0.0.1", "listen_port": 8790},
                 "edge": {"listen_host": "0.0.0.0", "listen_port": 8080}}

ALLOWED_POST = ("/v1/messages", "/v1/messages/count_tokens", "/v1/responses")

# Request headers never forwarded: hop-by-hop and framing (recomputed here), the client's
# credentials (the upstream holds the real ones), and Accept-Encoding, because the line-by-line
# stream relay needs an uncompressed body.
DROP_REQUEST = {"host", "connection", "keep-alive", "proxy-connection", "proxy-authorization", "te",
                "trailer", "transfer-encoding", "upgrade", "content-length", "expect", "accept-encoding",
                "authorization", "x-api-key", "cookie"}
# Response headers passed back. Everything else is dropped (the upstream's Server and its own x-* headers).
KEEP_RESPONSE = {"content-type", "retry-after", "cache-control"}

# Error text that would tell the agent what sits behind the gateway (default; leak_pattern replaces it).
LEAK = re.compile(r"x-cost|x-fwd|\bbackends?\b|\brouter\b|\bproxy\b|deployment|\bteam\b|forwarder|\bgate\b|\bedge\b", re.I)
ROUTER_PREFIX = None   # strip_error_prefix, compiled
# Fragments clients act on (Claude Code compacts on the first and resizes max_tokens on the
# second), kept when the rest of a leaking message is replaced.
ACTIONABLE = (re.compile(r"prompt is too long(?::\s*\d+ tokens > \d+ maximum)?", re.I),
              re.compile(r"input length and `?max_tokens`? exceed context limit: \d+ \+ \d+ > \d+", re.I))
GENERIC_BY_STATUS = {400: "Invalid request.", 401: "Authentication failed.", 403: "Permission denied.",
                     404: "Not found.", 408: "Request timed out.", 413: "Request too large.",
                     429: "Rate limited. Please retry later.", 500: "Internal server error.",
                     502: "Upstream service unavailable.", 503: "Service unavailable.",
                     504: "Upstream timed out.", 529: "Overloaded."}
GENERIC_BY_TYPE = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403,
                   "not_found_error": 404, "request_too_large": 413, "rate_limit_error": 429,
                   "api_error": 500, "overloaded_error": 529}

MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"\\]{1,200})"')
RID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")

CFG = {}
UPSTREAM = None      # (host, port, path prefix)
RUN_TAG = None       # compiled run_tag_re
LINK_KEY = None      # bytes from link_token_file
LOG_LOCK = threading.Lock()
DROP_PREFIXES = ("x-cost-", "x-fwd-")   # drop_header_prefixes, set in main
INJECTED = set()     # the upstream header names the gate sets itself: a client's copy is never forwarded


def ms_since(t0):
    return round((time.monotonic() - t0) * 1000, 1)


def write_log(rec):
    line = json.dumps(rec, separators=(",", ":")) + "\n"
    with LOG_LOCK, open(CFG["log_path"], "a") as f:
        f.write(line)


def one_line(v):
    """v if it is a usable single-line header value, else None."""
    return v if isinstance(v, str) and 0 < len(v) <= 256 and v == v.strip() \
        and not any(ord(c) < 32 or ord(c) == 127 for c in v) else None


def clean_message(msg, status, etype):
    """The message with router naming removed; actionable fragments survive."""
    if ROUTER_PREFIX is not None:
        msg = ROUTER_PREFIX.sub("", msg)
    if not LEAK.search(msg):
        return msg
    for pat in ACTIONABLE:
        m = pat.search(msg)
        if m:
            return m.group(0)
    return GENERIC_BY_STATUS.get(GENERIC_BY_TYPE.get(etype) or status, "Request failed.")


def scrub(obj, status):
    """Clean every error message inside obj in place; True if anything changed."""
    changed = False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "error" and isinstance(v, str):
                new = clean_message(v, status, None)
                if new != v:
                    obj[k], changed = new, True
            elif k == "error" and isinstance(v, dict) and isinstance(v.get("message"), str):
                new = clean_message(v["message"], status, v.get("type") or v.get("code"))
                if new != v["message"]:
                    v["message"], changed = new, True
            if isinstance(v, (dict, list)):
                changed = scrub(v, status) or changed
    elif isinstance(obj, list):
        for v in obj:
            changed = scrub(v, status) or changed
    return changed


def scrub_bytes(data, status):
    """(bytes, changed) for a JSON error body; bytes that are not JSON pass unchanged."""
    try:
        obj = json.loads(data)
    except ValueError:
        return data, False
    if not scrub(obj, status):
        return data, False
    return json.dumps(obj).encode(), True


def error_body(path, etype, message):
    """An error in the dialect of the endpoint the client called."""
    if path == "/v1/responses":
        return {"error": {"message": message, "type": etype, "param": None, "code": None}}
    return {"type": "error", "error": {"type": etype, "message": message}}


def models_body():
    data = [{"id": m, "object": "model", "type": "model", "display_name": m,
             "created": 0, "created_at": "2026-01-01T00:00:00Z",
             "owned_by": "anthropic" if m.startswith("claude") else "openai"} for m in CFG["models"]]
    return {"object": "list", "data": data, "has_more": False,
            "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}


def model_of(body):
    m = MODEL_RE.search(body[:4096])
    if m:
        return m.group(1).decode("utf-8", "replace")
    try:
        v = json.loads(body).get("model")
        return v if isinstance(v, str) else None
    except (ValueError, AttributeError):
        return None


def pin_for(model):
    """The upstream backend this model is pinned to, or None. Matched on the id with any [1m] and
    :effort suffix stripped, so gpt-6.1-sol:high is pinned like gpt-6.1-sol."""
    if not model or not CFG["pin_backend_by_model"]:
        return None
    base = model.removesuffix("[1m]").split(":", 1)[0]
    return CFG["pin_backend_by_model"].get(base)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 900                          # an idle keep-alive client connection is closed after this
    error_content_type = "application/json"
    error_message_format = '{"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid request."}}'

    def log_message(self, *args):          # the JSON log replaces the stdlib's stderr lines
        pass

    def send_response(self, code, message=None):
        """The stdlib version also sends a Server header naming Python; this one does not."""
        self.send_response_only(code, message)
        self.send_header("Date", self.date_time_string())

    def do_POST(self):
        self.route()

    do_GET = do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_POST

    def read_body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            parts = []
            while True:
                size = int(self.rfile.readline(65537).split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    def send_json(self, status, obj):
        blob = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(blob)
        return len(blob)

    def refuse(self, rec, path, status, etype, message, why):
        rec.update(status=status, refused=why, resp_bytes=self.send_json(status, error_body(path, etype, message)))

    def identity(self):
        """(run tag, session id, request id, inject headers) for this request, or None to refuse."""
        if CFG["mode"] == "edge":
            rid = uuid.uuid4().hex
            inject = [("X-Fwd-Run", CFG["cost_tag"]), ("X-Fwd-Session", CFG["session_id"]), ("X-Fwd-Request-Id", rid)]
            if LINK_KEY:
                inject.append(("X-Fwd-Key", LINK_KEY.decode()))
            return CFG["cost_tag"], CFG["session_id"], rid, inject
        if LINK_KEY and not hmac.compare_digest((self.headers.get("X-Fwd-Key") or "").encode(), LINK_KEY):
            return None
        run = one_line(self.headers.get("X-Fwd-Run"))
        if run is not None and not RUN_TAG.match(run):
            run = None
        session = one_line(self.headers.get("X-Fwd-Session")) if run else None
        rid = self.headers.get("X-Fwd-Request-Id") or ""
        rid = rid if RID_RE.match(rid) else None
        if not run:
            run, session = CFG["cost_tag"], CFG["session_id"]
        if not run:
            return None
        H = CFG["upstream_headers"]
        inject = [(H["cost_tag"], run)] if H.get("cost_tag") else []
        if H.get("priority") and CFG["priority"]:
            inject.append((H["priority"], CFG["priority"]))
        if session and H.get("session"):
            inject.append((H["session"], session))
        if CFG["max_wait_s"] is not None and H.get("max_wait"):
            inject.append((H["max_wait"], str(CFG["max_wait_s"])))
        return run, session, rid, inject

    def stopped(self, run):
        if CFG["stop_file"] and os.path.exists(CFG["stop_file"]):
            return True
        return bool(CFG["stop_dir"] and run and os.path.exists(os.path.join(CFG["stop_dir"], run)))

    def route(self):
        wall, t0 = time.time(), time.monotonic()
        u = urllib.parse.urlsplit(self.path)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(wall)) + f".{int(wall % 1 * 1000):03d}Z",
               "ts_epoch": round(wall, 3), "mode": CFG["mode"], "method": self.command, "path": self.path[:512]}
        try:
            try:
                body = self.read_body()
            except (OSError, ValueError):
                self.close_connection = True
                rec.update(status=400, error="unreadable request body")
                return
            rec["req_bytes"], rec["body_ms"] = len(body), ms_since(t0)
            t_body = time.monotonic()
            if self.command == "GET" and u.path == "/v1/models" and CFG["models"]:
                rec.update(status=200, resp_bytes=self.send_json(200, models_body()))
                return
            if self.command != "POST" or u.path not in ALLOWED_POST:
                return self.refuse(rec, u.path, 404, "not_found_error", "Not found.", "not_allowed")
            ident = self.identity()
            if ident is None:
                return self.refuse(rec, u.path, 403, "permission_error", "Permission denied.", "no_run_tag_or_key")
            run, session, rid, inject = ident
            rec.update(run=run, session=session, rid=rid)
            model = model_of(body)
            rec["model"] = model
            if CFG["allowed_models"] and (model or "").removesuffix("[1m]") not in CFG["allowed_models"]:
                return self.refuse(rec, u.path, 404, "not_found_error", f"model: {model}", "model_not_allowed")
            if self.stopped(run):
                rec["forwarder_wait_ms"] = ms_since(t_body)
                return self.refuse(rec, u.path, 429, "rate_limit_error", GENERIC_BY_STATUS[429], "stopped")
            pin = pin_for(model)             # load_config allows pins on a gate only
            if pin:
                inject.append((CFG["upstream_headers"]["backend"], pin))
                rec["pinned"] = pin
            self.forward(u, body, inject, rec, t0, t_body)
        except (BrokenPipeError, ConnectionResetError):
            rec["client_gone"] = True
            self.close_connection = True
        finally:
            rec["total_ms"] = ms_since(t0)
            write_log(rec)

    def forward(self, u, body, inject, rec, t0, t_body):
        host, port, prefix = UPSTREAM
        up = http.client.HTTPConnection(host, port, timeout=CFG["upstream_timeout_s"])
        try:
            up.putrequest("POST", prefix + self.path, skip_host=True, skip_accept_encoding=True)
            up.putheader("Host", f"{host}:{port}")
            for k, v in self.headers.items():
                kl = k.lower()
                if kl not in DROP_REQUEST and not kl.startswith(DROP_PREFIXES) and kl not in INJECTED:
                    up.putheader(k, v)
            for k, v in inject:
                up.putheader(k, v)
            up.putheader("Content-Length", str(len(body)))
            up.endheaders(body)
            rec["forwarder_wait_ms"] = ms_since(t_body)
            resp = up.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            up.close()
            rec.update(error=f"upstream: {type(exc).__name__}")
            rec.setdefault("forwarder_wait_ms", ms_since(t_body))
            rec.update(status=502, resp_bytes=self.send_json(502, error_body(u.path, "api_error",
                                                                              GENERIC_BY_STATUS[502])))
            return
        rec["upstream_headers_ms"] = ms_since(t0)
        rec["status"] = resp.status
        ctype = resp.getheader("Content-Type") or ""
        keep = [(k, v) for k, v in resp.getheaders() if k.lower() in KEEP_RESPONSE]
        try:
            if ctype.split(";")[0].strip().lower() == "text/event-stream":
                rec["sse"] = True
                self.relay_stream(resp, keep, rec, t0)
            else:
                data = resp.read()
                if resp.status >= 400 and CFG["scrub_errors"]:
                    data, rec["scrubbed"] = scrub_bytes(data, resp.status)
                self.send_response(resp.status)
                for k, v in keep:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                rec["first_byte_ms"], rec["resp_bytes"] = ms_since(t0), len(data)
        except (BrokenPipeError, ConnectionResetError):
            rec["client_gone"] = True        # closing upstream below tells the next hop the client left
            self.close_connection = True
        except (OSError, http.client.HTTPException) as exc:
            rec["error"] = f"upstream mid-response: {type(exc).__name__}"
            self.close_connection = True
        finally:
            up.close()

    def relay_stream(self, resp, keep, rec, t0):
        """SSE, one line at a time as it arrives. Only error-event data lines can change."""
        self.send_response(resp.status)
        for k, v in keep:
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        event, sent = None, 0
        while True:
            line = resp.readline()
            if not line:
                break
            if line.startswith(b"event:"):
                event = line[6:].strip()
                if event != b"ping" and "first_event_ms" not in rec:
                    rec["first_event_ms"] = ms_since(t0)     # the first byte may be a keep-alive ping
            elif line.startswith(b"data:") and event in (b"error", b"response.failed") and CFG["scrub_errors"]:
                payload, changed = scrub_bytes(line[5:].strip(), 200)
                if changed:
                    line, rec["scrubbed"] = b"data: " + payload + b"\n", True
            elif not line.strip():
                event = None
            self.wfile.write(line)
            if not sent:
                self.wfile.flush()
                rec["first_byte_ms"] = ms_since(t0)
            sent += len(line)
        rec["resp_bytes"] = sent


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        sys.exit(f"unknown config keys: {sorted(unknown)}")
    mode = raw.get("mode")
    if mode not in MODE_DEFAULTS:
        sys.exit('mode must be "gate" or "edge"')
    cfg = {**DEFAULTS, **{k: v for k, v in MODE_DEFAULTS[mode].items()}, **raw}
    cfg["log_path"] = cfg["log_path"] or f"{mode}.jsonl"
    for key in ("cost_tag", "session_id", "priority"):
        if cfg[key] is not None and one_line(cfg[key]) is None:
            sys.exit(f"{key} must be a non-empty single-line string of at most 256 characters")
    if cfg["cost_tag"] and not re.match(cfg["run_tag_re"], cfg["cost_tag"]):
        sys.exit("cost_tag does not match run_tag_re")
    if mode == "edge":
        if not cfg["cost_tag"]:
            sys.exit("an edge needs cost_tag: without it the run's spend cannot be found in the ledger")
        if not cfg["upstream"]:
            sys.exit("an edge needs upstream: the gate's address as seen from this host")
        if cfg["stop_dir"] or cfg["max_wait_s"] is not None:
            sys.exit("stop_dir and max_wait_s are gate settings")
    if cfg["cost_tag"]:
        cfg["session_id"] = cfg["session_id"] or str(uuid.uuid4())
    elif cfg["session_id"]:
        sys.exit("session_id without cost_tag has no effect: the edge supplies sessions")
    if mode == "gate" and not cfg["upstream"]:
        sys.exit("a gate needs upstream: the model router or API it forwards to")
    hdrs = {**DEFAULTS["upstream_headers"], **(raw.get("upstream_headers") or {})}
    if not all(v is None or (isinstance(v, str) and re.match(r"^[A-Za-z0-9-]{1,64}$", v)) for v in hdrs.values()):
        sys.exit("upstream_headers values must be header names or null")
    cfg["upstream_headers"] = hdrs
    pins = cfg["pin_backend_by_model"]
    if not isinstance(pins, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) and re.match(r"^[a-z][a-z0-9_-]{0,31}$", v)
            for k, v in pins.items()):
        sys.exit("pin_backend_by_model must map model ids to upstream backend names, e.g. {\"some-model\": \"backend-a\"}")
    if pins and mode != "gate":
        sys.exit("pin_backend_by_model is a gate setting: the gate strips the client's headers from the edge")
    if pins and not hdrs.get("backend"):
        sys.exit("pin_backend_by_model needs upstream_headers.backend: the header the upstream reads a pin from")
    return cfg


def main():
    global CFG, UPSTREAM, RUN_TAG, LINK_KEY, LEAK, ROUTER_PREFIX, DROP_PREFIXES, INJECTED
    if len(sys.argv) != 2:
        sys.exit("usage: forwarder.py <config.json>")
    CFG = load_config(sys.argv[1])
    if CFG["leak_pattern"]:
        LEAK = re.compile(CFG["leak_pattern"], re.I)
    if CFG["strip_error_prefix"]:
        ROUTER_PREFIX = re.compile(CFG["strip_error_prefix"])
    INJECTED = {v.lower() for v in CFG["upstream_headers"].values() if v}
    DROP_PREFIXES = tuple(p.lower() for p in CFG["drop_header_prefixes"])
    RUN_TAG = re.compile(CFG["run_tag_re"])
    if CFG["link_token_file"]:
        with open(os.path.expanduser(CFG["link_token_file"]), "rb") as f:
            LINK_KEY = f.read().strip()
        if not LINK_KEY or any(c < 33 or c > 126 for c in LINK_KEY):
            sys.exit("link_token_file must hold one line of printable ASCII")
    up = urllib.parse.urlsplit(CFG["upstream"])
    if up.scheme != "http" or not up.hostname:
        sys.exit("upstream must be an http:// URL")
    UPSTREAM = (up.hostname, up.port or 80, up.path.rstrip("/"))
    server = http.server.ThreadingHTTPServer((CFG["listen_host"], CFG["listen_port"]), Handler)
    server.daemon_threads = True
    write_log({"event": "start", "ts_epoch": round(time.time(), 3), "mode": CFG["mode"],
               "run": CFG["cost_tag"], "session": CFG["session_id"], "upstream": CFG["upstream"],
               "listen": f'{CFG["listen_host"]}:{CFG["listen_port"]}', "allowed_models": CFG["allowed_models"],
               "stop_file": CFG["stop_file"], "stop_dir": CFG["stop_dir"], "link_key": bool(LINK_KEY),
               "pin_backend_by_model": CFG["pin_backend_by_model"]})
    server.serve_forever()


if __name__ == "__main__":
    main()
