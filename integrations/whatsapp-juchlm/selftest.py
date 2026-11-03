"""Self-test del puente (sin tocar OpenWA ni ChatJUCHLM reales).
Firma y envia en el MISMO proceso para eliminar quirks de PowerShell."""
import hashlib
import hmac
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent
secret = None
for line in (BASE / ".env").read_text(encoding="utf-8-sig").splitlines():
    if line.startswith("WEBHOOK_SECRET="):
        secret = line.split("=", 1)[1].strip()
assert secret, "no secret"

URL = "http://127.0.0.1:2786/webhook"


def post(body: bytes, sig: str) -> tuple[int, str]:
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if sig:
        req.add_header("X-OpenWA-Signature", sig)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def sign(b: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), b, hashlib.sha256).hexdigest()


msg = {
    "event": "message.received",
    "sessionId": "s",
    "idempotencyKey": "probe-1",
    "deliveryId": "d-1",
    "timestamp": 1,
    "data": {
        "chatId": "51900000000@c.us",
        "from": "51900000000@c.us",
        "fromMe": True,     # fromMe -> no debe activar pipeline
        "isGroup": False,
        "type": "chat",
        "body": "ping de prueba",
        "timestamp": 1,
    },
}
body = json.dumps(msg, ensure_ascii=False).encode("utf-8")

# health
import urllib.error
with urllib.request.urlopen("http://127.0.0.1:2786/healthz", timeout=10) as r:
    print("HEALTHZ:", r.status, r.read().decode())

c1, _ = post(body, sign(body))
print("1) HMAC valido + fromMe=true ->", c1, "(esperado 200, sin pipeline)")

wrong = "sha256=" + "0" * 64
c2, _ = post(body, wrong)
print("2) firma invalida ->", c2, "(esperado 403)")

c3, _ = post(body, sign(body))
print("3) replay mismo key ->", c3, "(esperado 200, dedup)")

# caso real con pipeline pero chatId de prueba 51999999999 y mensaje 'hola'
# (con SEND_ACK iria a OpenWA, que no esta arriba: wa_send registrara fallo en log;
#  comprobamos que pipeline no reviente el handler: la respuesta webhook ya es 200 ACK)
msg2 = dict(msg)
msg2["data"] = dict(msg["data"], fromMe=False, body="hola", idempotencyKey=None)
msg2["idempotencyKey"] = "probe-2"
msg2["deliveryId"] = "d-2"
b2 = json.dumps(msg2, ensure_ascii=False).encode("utf-8")
c4, _ = post(b2, sign(b2))
print("4) mensaje real de prueba ->", c4, "(esperado 200 ACK async; wa_send a OpenWA fallara, ok)")

print("Nota: el fallo de envio a OpenWA en el caso 4 es esperado (gateway apagado);")
print("confirma que su evento/pipeline no rompe el handler.")
