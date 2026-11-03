# Probe: OpenWA -> ChatJUCHLM (WhatsApp)

> **Estado:** Fase de PRUEBA aislada. El sistema web (`app.py`, plantillas, .venv, chroma_db, SQL)
> está **intacto**. Todo este contenido vive en `integrations/` (gitignoreada).

## Componentes

| Componente | Puerto | Qué es |
|---|---|---|
| ChatJUCHLM (FASTAPI existente) | 8000 | **No se modifica.** Solo se consulta `POST /api/conversation/message` |
| OpenWA (gateway WA, NestJS) | 2785 (API+dashboard) / 2886 (dashboard dev) | `integrations/OpenWA/` — código ajeno (shallow clone) |
| Puente (este proyecto chico) | 2786 | `bridge.py` — stdlib Python, webhook receiver + forwarder |

## Flujo

1. OpenWA recibe mensajes de la sesión emparejada (QR) → dispara webhook `message.received`
   firmado con HMAC (`X-OpenWA-Signature: sha256=...`).
2. El puente valida la firma, deduplica (`idempotencyKey`), ignora `fromMe`/grupos (por defecto)
   y multimedia (responde amable que solo procesa texto en la prueba).
3. Cada chat `519xxxx@c.us` ⇒ sesión FSM estable `wa519xxxx...`:
   - Primer mensaje del chat → `POST /api/conversation/reset` (inicializa sin gastar turno)
   - Cada texto → `POST /api/conversation/message` → respuesta `message` + `actions`.
4. El puente convierte la respuesta a formato WhatsApp (negritas, listas, máx 3800 chars)
   y la manda con `POST /api/sessions/{sesion}/messages/send-text` (API key).

## Puesta en marcha (Fase 0 → 3)

```powershell
# Fase 0 (una vez): OpenWA
cd integrations/OpenWA
npm ci          # hecho
npm run dev     # API :2785 + dashboard :2886  (dejar corriendo)
# La primera imagen del boot imprime la API key admin y la deja en data/.api-key

# Crear la sesion de prueba + persistencia del QR en el dashboard
#   Dashboard: http://localhost:2886 (login con la API key)  ->  Session "juchlm-probe"
#   o por API:
$K = Get-Content .\integrations\OpenWA\data\.api-key
curl -X POST http://localhost:2785/api/sessions -H "X-API-Key: $K" -d '{"name":"juchlm-probe"}'
curl -X POST http://localhost:2785/api/sessions/juchlm-probe/start -H "X-API-Key: $K"
curl http://localhost:2785/api/sessions/juchlm-probe/qr -H "X-API-Key: $K"   # escanear con WhatsApp
# NOTA (validado): los endpoints de sesion operan con el UUID, no con el nombre;
# heredado de este probe: 3f0414ed-c527-4694-8967-60f0a556c2a1 (queda en OPENWA_SESSION del .env del puente).
# Tras reiniciar OpenWA la sesion amanece 'disconnected': con POST /api/sessions/{uuid}/start
# se reconecta con las credenciales guardadas en data/sessions/ (NO hace falta re-escanear QR).
```

```powershell
# Fase 3: el puente (en otra terminal, desde la raiz del proyecto)
Copy-Item integrations\whatsapp-juchlm\.env.example integrations\whatsapp-juchlm\.env
notepad integrations\whatsapp-juchlm\.env   # WEBHOOK_SECRET + OPENWA_API_KEY
.\.venv\Scripts\python.exe integrations\whatsapp-juchlm\bridge.py

# Requisito (validado): permitir webhook hacia el puente local en integrations\OpenWA\.env
#   SSRF_ALLOWED_HOSTS=127.0.0.1,localhost     (escape oficial: src/common/security/ssrf-guard.ts)
#   Se aplica al reiniciar OpenWA. Sin esto: 400 {"message":"Destination address is not allowed"}.

# Registrar el webhook en OpenWA (una vez, con el puente ya arriba):
$BR = "http://127.0.0.1:2786/webhook"
$SID = "3f0414ed-c527-4694-8967-60f0a556c2a1"   # uuid de juchlm-probe (ver OPENWA_SESSION)
$SECRET = (Get-Content integrations\whatsapp-juchlm\.env | Where-Object { $_ -like 'WEBHOOK_SECRET=*' }).Split('=')[1]
curl -X POST http://localhost:2785/api/sessions/$SID/webhooks `
  -H "X-API-Key: $K" -H "Content-Type: application/json" `
  -d "{\"url\":\"$BR\",\"events\":[\"message.received\"],\"secret\":\"$SECRET\"}"
# Verificar: curl http://localhost:2785/api/sessions/$SID/webhooks -H "X-API-Key: $K"
# (el webhook persiste en SQLite; tras reiniciar OpenWA re-verificar y re-crear si no aparece)
```

## Prueba de aceptación (manual)

Desde el celular del número dedicado intercambiar mensajes con el número emparejado:

1. "hola" → menú principal del FSM
2. "explicame que es la junta de usuarios" → respuesta RAG con evidencia
3. "consulta mi predio" → flujo FSM pide DNI → responde datos (si SQL Server activo)
4. Multimedia (una imagen) → respuesta: solo texto en la prueba

## Seguridad y riesgos (repetir de 'README de OpenWA')

- **Número dedicado**: jamás conectar un número personal o el institucional principal.
  Riesgo de restricción/ban no nulo (gateway no oficial). Hoy solo usamos respuestas
  (reply traffic = el tráfego más seguro).
- Webhook firmado + dedup; puente escucha **solo en 127.0.0.1**.
- `warm-up` del número antes de tráfico real, y `SIMULATE_TYPING` por defecto ya ayuda.

## Validación end-to-end (2026-11-02) — RESULTADO: OK

Prueba real: mensaje "hola" desde otro celular → respuesta del asistente de vuelta por WhatsApp.

- Contracto corregido: OpenWA **normaliza** wwebjs: `chat` → `text`
  (`src/engine/adapters/message-mapper.ts`); respuestas de botón/lista también llegan como `text`
  (la elección viene en `body`). El puente acepta ambos; tipos de sistema sin body (`revoked`,
  `unknown`) se ignoran en silencio; multimedia recibe aviso de "solo texto".
- `chatId` puede llegar como LID (p. ej. `85878424760498@lid`): el puente deriva el sid de los
  dígitos ⇒ FSM igual de correcto (`wa85878424760498`).
- Latencia de la primera consulta (modelos en frío): ~2 min 06 s — bajo `JC_TIMEOUT=150`.
  Consultas en caliente van a ser mucho más rápido.

Traza del log del puente (el "sello" del circuito):

```
12:19:24 webhook event=message.received key=msg_3f0414ed-...   (HMAC ok)
12:19:24 chat nuevo 85878424760498@lid -> inicializar sesion FSM wa85878424760498
12:19:27 wa_send ok 85878424760498@lid (36 chars)              # ACK 'Consultando...'
12:19:27 jc <- 'Hola'  sid=wa85878424760498                    # -> ChatJUCHLM (web system limpio)
12:21:31 jc -> state=RAG_RESPONSE 1498 chars
12:21:36 wa_send ok 85878424760498@lid (1498 chars)            # respuesta en WhatsApp
```

Pendientes de la lista de aceptación: flujo FSM "consulta mi predio" (requiere SQL Server
activo) y prueba de multimedia con el aviso educado.

## Paridad con el portal en WhatsApp (v2 del puente)

La experiencia del chat web (botones, PDF) se replica en el canal por el mismo contrato HTTP:

1. **Botones como opciones numeradas.** Cuando la respuesta del FSM trae >= 2 acciones y el
   texto NO contiene ya su propio menú numerado, el puente renderiza:
   `1. 📚 Información de la Junta / 2. 🏠 Información del predio` — y responder el número
   dispara el mismo click del portal (`POST /api/conversation/message` con `action_id`).
2. **No-robo de dígitos.** Si el FSM ya numeró su menú en el texto (lista de predios del
   padrón), el puente no numera ni intercepta: los dígitos siguen al FSM como texto. Con
   un solo botón (p. ej. "⬅️ Volver al menú") se sugiere escribir su etiqueta, sin número.
3. **PDF como documento adjunto.** Las respuestas con `data.download_url`
   (constancia de trámite) se descargan del portal y se entregan con
   `POST /messages/send-document` (base64, mimetype, filename + caption). Si la descarga
   falla o supera 25 MB, el mensaje conserva el enlace absoluto al portal.
4. Clicks y consultas usan la MISMA sesión FSM (`wa<digits>`) que el portal asigna por
   `session_id`; el PDF queda ligado a esa sesión (`?sid=<sid>`).
5. **Indicador de escritura.** Tras el mensaje "Consultando al asistente…" el puente enciende
   "escribiendo…" en el chat y lo mantiene vivo mientras el asistente procesa (RAG/FSM, hasta
   ~2 min con modelos en frío): un re-afirmado a los 3 s (por si el envío del ACK limpió el
   indicador en WhatsApp Web) y keep-alive cada 20 s (WhatsApp lo vence a ~25 s). Al entregar
   la respuesta lo limpia con `paused`. Desactivable con `SEND_TYPING=false`.

## Rollback (cero rastro)

- Ctrl+C en OpenWA y en el puente.
- `Remove-Item -Recurse -Force integrations` + quitar la línea `integrations/` de `.gitignore`.
- El sistema web no recibió ningún cambio ⇒ nada que revertir allí.
