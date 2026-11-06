<#
    Lanzador de un clic del stack JUCHLM (web + OpenWA + puente WhatsApp).

    Por que existe: cada reinicio de la maquina (o del harness) deja al bot
    "mudo" porque hay que relanzar 3 procesos y reconectar la sesion de WhatsApp
    a mano. Este script hace todo eso y verifica que quedo en pie.

    Uso (PowerShell):
        .\integrations\whatsapp-juchlm\start_stack.ps1

    Es idempotente: si un servicio ya esta escuchando, no lo relanza.
    Para detener: .\integrations\whatsapp-juchlm\stop_stack.ps1
#>
[CmdletBinding()]
param(
    [int]$EsperaOpenWA = 300,      # segundos maximos para que arranque OpenWA
    [int]$EsperaSesion = 180,      # segundos maximos para que la sesion quede 'ready'
    [switch]$SinPuente,             # no levantar el puente de WhatsApp
    [switch]$SinSesion              # no tocar la sesion de WhatsApp
)

$ErrorActionPreference = 'Continue'
$raiz = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dirOpenWA = Join-Path $raiz 'integrations\OpenWA'
$dirPuente = Join-Path $raiz 'integrations\whatsapp-juchlm'
$logDir = Join-Path $dirPuente 'logs'
$venvPy = Join-Path $raiz '.venv\Scripts\python.exe'

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

function Write-Paso([string]$txt) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $txt)
}

function Test-Puerto([int]$puerto) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect('127.0.0.1', $puerto, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1200, $false) -and $c.Connected
        $c.Close()
        return $ok
    } catch { return $false }
}

function Test-Http([string]$url, [int]$segundos = 5) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing $url -TimeoutSec $segundos
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400)
    } catch { return $false }
}

function Espera-Http([string]$url, [int]$maxSegundos, [string]$etiqueta) {
    $limite = (Get-Date).AddSeconds($maxSegundos)
    while ((Get-Date) -lt $limite) {
        if (Test-Http $url) { Write-Paso "$etiqueta listo"; return $true }
        Start-Sleep -Seconds 3
    }
    Write-Host "[AVISO] $etiqueta no respondio en ${maxSegundos}s ($url)" -ForegroundColor Yellow
    return $false
}

function Leer-Env([string]$ruta) {
    $tabla = @{}
    if (-not (Test-Path $ruta)) { return $tabla }
    foreach ($linea in Get-Content $ruta) {
        if ($linea -match '^\s*#' -or -not $linea.Contains('=')) { continue }
        $k, $v = $linea -split '=', 2
        $tabla[$k.Trim()] = $v.Trim()
    }
    return $tabla
}

function Iniciar-Proceso([string]$nombre, [string]$exe, [string[]]$argumentos, [string]$dir, [string]$logBase) {
    Write-Paso "Levantando $nombre ..."
    $out = Join-Path $logDir "$logBase.out.log"
    $err = Join-Path $logDir "$logBase.err.log"
    Start-Process -FilePath $exe -ArgumentList $argumentos -WorkingDirectory $dir `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
}

# ---------------------------------------------------------------- 1) portal web
if (Test-Puerto 8000) {
    Write-Paso 'Portal ChatJUCHLM (:8000) ya estaba en pie'
} else {
    if (-not (Test-Path $venvPy)) {
        Write-Host "[ERROR] No existe $venvPy (crea el entorno: python -m venv .venv)" -ForegroundColor Red
        exit 1
    }
    Iniciar-Proceso 'portal ChatJUCHLM' $venvPy @('app.py') $raiz 'portal'
    Espera-Http 'http://127.0.0.1:8000/health' 120 'Portal ChatJUCHLM' | Out-Null
}

# ---------------------------------------------------------------- 2) OpenWA
if (Test-Puerto 2785) {
    Write-Paso 'OpenWA (:2785) ya estaba en pie'
} else {
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
    if (-not $npm) { $npm = (Get-Command npm -ErrorAction SilentlyContinue) }
    if (-not $npm) {
        Write-Host '[ERROR] npm no esta en el PATH (Node.js requerido para OpenWA)' -ForegroundColor Red
    } else {
        Iniciar-Proceso 'OpenWA' $npm.Source @('run', 'dev') $dirOpenWA 'openwa'
        Espera-Http 'http://127.0.0.1:2785/api/health' $EsperaOpenWA 'OpenWA' | Out-Null
    }
}

# ---------------------------------------------------------------- 3) sesion WA
$envPuente = Leer-Env (Join-Path $dirPuente '.env')
$openwaUrl = if ($envPuente['OPENWA_URL']) { $envPuente['OPENWA_URL'] } else { 'http://127.0.0.1:2785' }
$sesionId = $envPuente['OPENWA_SESSION']
$apiKey = $envPuente['OPENWA_API_KEY']
$secret = $envPuente['WEBHOOK_SECRET']
$urlHook = 'http://127.0.0.1:2786/webhook'

if (-not $SinSesion -and $sesionId -and $apiKey -and (Test-Http "$openwaUrl/api/health")) {
    $cabeceras = @{ 'X-API-Key' = $apiKey }
    try {
        $estado = (Invoke-RestMethod "$openwaUrl/api/sessions/$sesionId" -Headers $cabeceras -TimeoutSec 15).status
        if ($estado -ne 'ready') {
            Write-Paso "Sesion '$sesionId' esta '$estado': reconectando ..."
            try {
                Invoke-RestMethod "$openwaUrl/api/sessions/$sesionId/start" -Method Post -Headers $cabeceras `
                    -ContentType 'application/json' -Body '{}' -TimeoutSec 120 | Out-Null
            } catch { Write-Host "[AVISO] start fallo: $($_.Exception.Message)" -ForegroundColor Yellow }
            $limite = (Get-Date).AddSeconds($EsperaSesion)
            while ((Get-Date) -lt $limite) {
                Start-Sleep -Seconds 5
                $estado = (Invoke-RestMethod "$openwaUrl/api/sessions/$sesionId" -Headers $cabeceras -TimeoutSec 15).status
                if ($estado -eq 'ready') { break }
            }
        }
        Write-Paso "Sesion WhatsApp: $estado"

        # webhook: si no esta (o no apunta al puente) se registra con el secreto HMAC
        $hooks = @(Invoke-RestMethod "$openwaUrl/api/sessions/$sesionId/webhooks" -Headers $cabeceras -TimeoutSec 15)
        if (-not ($hooks | Where-Object { $_.url -eq $urlHook })) {
            Write-Paso "Registrando webhook $urlHook"
            $cuerpo = @{ url = $urlHook; events = @('message.received') }
            if ($secret) { $cuerpo['secret'] = $secret }
            Invoke-RestMethod "$openwaUrl/api/sessions/$sesionId/webhooks" -Method Post -Headers $cabeceras `
                -ContentType 'application/json' -Body (ConvertTo-Json $cuerpo) -TimeoutSec 30 | Out-Null
            Write-Paso 'Webhook registrado'
        } else {
            Write-Paso 'Webhook ya registrado y activo'
        }
    } catch {
        Write-Host "[AVISO] No se pudo verificar la sesion: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- 4) puente
if ($SinPuente) {
    Write-Paso 'Puente WhatsApp omitido (-SinPuente)'
} elseif (Test-Puerto 2786) {
    Write-Paso 'Puente WhatsApp (:2786) ya estaba en pie'
} else {
    Iniciar-Proceso 'puente WhatsApp' $venvPy @('integrations\whatsapp-juchlm\bridge.py') $raiz 'puente'
    Espera-Http 'http://127.0.0.1:2786/healthz' 60 'Puente WhatsApp' | Out-Null
}

# ---------------------------------------------------------------- resumen
Write-Host ''
Write-Host '=== ESTADO DEL STACK JUCHLM ===' -ForegroundColor Cyan
foreach ($fila in @(
        @{ n = 'Portal ChatJUCHLM'; u = 'http://127.0.0.1:8000/health' },
        @{ n = 'OpenWA (gateway WA) '; u = 'http://127.0.0.1:2785/api/health' },
        @{ n = 'Puente WhatsApp    '; u = 'http://127.0.0.1:2786/healthz' })) {
    $estado = if (Test-Http $fila.u) { 'OK' } else { 'CAIDO' }
    $color = if ($estado -eq 'OK') { 'Green' } else { 'Red' }
    Write-Host ("  {0} : {1}" -f $fila.n, $estado) -ForegroundColor $color
}
Write-Host "  Logs: $logDir" -ForegroundColor DarkGray
Write-Host ''
