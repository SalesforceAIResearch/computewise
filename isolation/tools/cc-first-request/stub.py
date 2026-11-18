"""Local stand-in for the inference route, for checking what Claude Code sends. No model is called.

It applies the one rule observed on a provider's Anthropic-compatible path on 2026-10-20: a request whose
thinking.display is set to anything but 'summarized' or 'omitted' is refused. As the gateway did, a
streamed request is refused inside an HTTP 200 event stream, and a non-streamed one with HTTP 400.
Anything else on /v1/messages gets a minimal successful reply. Every request is logged to --out.
"""
import argparse, json, http.server

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ERR = {"type": "error", "error": {"type": "invalid_request_error",
       "message": "thinking.adaptive.display: Input should be 'summarized', 'omitted'"}}
MSG = {"id": "msg_stub", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
       "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
       "usage": {"input_tokens": 1, "output_tokens": 1}}
SSE = [("message_start", {"type": "message_start", "message": {**MSG, "content": [], "stop_reason": None,
                                                               "usage": {"input_tokens": 1, "output_tokens": 0}}}),
       ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
       ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
       ("content_block_stop", {"type": "content_block_stop", "index": 0}),
       ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                          "usage": {"output_tokens": 1}}),
       ("message_stop", {"type": "message_stop"})]
NOT_FOUND = b'{"type":"error","error":{"type":"not_found_error","message":"not found"}}'


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def record(self, body, outcome):
        rec = {"method": self.command, "path": self.path, "beta": self.headers.get("anthropic-beta"),
               "len": len(body) if body is not None else None, "outcome": outcome}
        if body:
            try:
                j = json.loads(body)
                rec.update(stream=j.get("stream"), thinking=j.get("thinking"), model=j.get("model"))
            except Exception as e:
                rec["parse_error"] = str(e)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def do_GET(self):
        self.record(None, "404")
        self.reply(404, "application/json", NOT_FOUND)

    do_HEAD = do_GET

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            j = json.loads(body) if body else {}
        except Exception:
            j = {}
        if not self.path.startswith("/v1/messages") or "count_tokens" in self.path:
            self.record(body, "404")
            return self.reply(404, "application/json", NOT_FOUND)
        disp = (j.get("thinking") or {}).get("display")
        bad = disp is not None and disp not in ("summarized", "omitted")
        if bad and j.get("stream"):
            self.record(body, "200-sse-error")
            return self.reply(200, "text/event-stream", f"event: error\ndata: {json.dumps(ERR)}\n\n".encode())
        if bad:
            self.record(body, "400")
            return self.reply(400, "application/json", json.dumps(ERR).encode())
        events, msg = SSE, MSG
        if disp == "summarized":  # answer as the API does with a summarised thinking block before the text
            think = [("content_block_start", {"type": "content_block_start", "index": 0,
                                              "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
                     ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                              "delta": {"type": "thinking_delta", "thinking": "STUB-SUMMARY: deciding to reply ok"}}),
                     ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                              "delta": {"type": "signature_delta", "signature": "c3R1Yg=="}}),
                     ("content_block_stop", {"type": "content_block_stop", "index": 0})]
            shifted = []
            for e, d in SSE[1:]:
                d = dict(d)
                if "index" in d:
                    d["index"] = 1
                shifted.append((e, d))
            events = [SSE[0]] + think + shifted
            msg = {**MSG, "content": [{"type": "thinking", "thinking": "STUB-SUMMARY: deciding to reply ok",
                                       "signature": "c3R1Yg=="}] + MSG["content"]}
        if j.get("stream"):
            self.record(body, "200-sse")
            return self.reply(200, "text/event-stream",
                              "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode())
        self.record(body, "200")
        return self.reply(200, "application/json", json.dumps(msg).encode())


http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
