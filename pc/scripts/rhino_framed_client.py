"""Minimal rhinomcp 0.3.2 wire client: connects through the SSH tunnel
(Mac 127.0.0.1:2999 -> PC 127.0.0.1:1999) and asks the live Rhino plugin for
real document info. Speaks the 0.3.2 framed protocol (4-byte big-endian length
header + UTF-8 JSON), with a fallback path if the plugin replies unframed."""
import socket, json, struct, sys

HOST, PORT = "127.0.0.1", 2999
cmd = {"type": "get_document_info", "params": {}}

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(20)
s.connect((HOST, PORT))

payload = json.dumps(cmd).encode("utf-8")
s.sendall(len(payload).to_bytes(4, "big") + payload)   # framed request


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            break
        buf += chunk
    return buf


header = recv_exact(s, 4)
if len(header) < 4:
    print("NO_RESPONSE (got %d bytes)" % len(header)); sys.exit(2)

if header[:1] == b"{":
    # Unframed legacy reply: keep reading until socket idle
    data = header
    s.settimeout(3)
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    print("UNFRAMED_REPLY:")
    print(data.decode("utf-8", "replace")[:2000])
else:
    length = int.from_bytes(header, "big")
    body = recv_exact(s, length)
    print("FRAMED_REPLY (%d bytes):" % length)
    try:
        print(json.dumps(json.loads(body.decode("utf-8")), indent=2)[:2500])
    except Exception:
        print(body.decode("utf-8", "replace")[:2000])
s.close()
