"""Stand-in for the rhinomcp Rhino plugin: a loopback TCP server on
127.0.0.1:1999 that speaks the 0.2.1 UNFRAMED protocol (raw JSON in, raw JSON
out). Proves the SSH tunnel + client wiring end-to-end without needing Rhino.
Runs for a bounded time then exits."""
import socket, json, time, sys

HOST, PORT = "127.0.0.1", 1999
DEADLINE = time.time() + 90

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))          # loopback ONLY, never 0.0.0.0
srv.listen(5)
srv.settimeout(2.0)
print(f"STUB_LISTENING {HOST}:{PORT}", flush=True)

while time.time() < DEADLINE:
    try:
        conn, addr = srv.accept()
    except socket.timeout:
        continue
    with conn:
        conn.settimeout(5.0)
        try:
            data = conn.recv(65536)
        except socket.timeout:
            data = b""
        try:
            req = json.loads(data.decode("utf-8")) if data else {}
        except Exception:
            req = {"raw": data.decode("utf-8", "replace")}
        # Mimic a rhinomcp response envelope (unframed: raw JSON, no length header)
        resp = {
            "status": "success",
            "result": {
                "served_by": "pc_stub_listener",
                "machine": "windows-host.example",
                "peer": str(addr),
                "received_command": req.get("type") or req.get("command"),
                "note": "tunnel + client env wiring proven with real bytes",
            },
        }
        conn.sendall(json.dumps(resp).encode("utf-8"))
        print(f"STUB_SERVED {req.get('type') or req.get('command')}", flush=True)
print("STUB_EXIT", flush=True)
