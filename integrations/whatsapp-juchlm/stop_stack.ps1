<#
    Detiene el stack JUCHLM (portal + OpenWA + puente WhatsApp).

    Util cuando hay que reiniciar por completo (por ejemplo tras un cambio en
    OpenWA o cuando la sesion de WhatsApp queda trabada).

    Uso:
        .\integrations\whatsapp-juchlm\stop_stack.ps1
#>
[CmdletBinding()]
param(
    [switch]$SoloPuente,
    [switch]$SoloOpenWA,
    [switch]$SoloPortal
)

$ErrorActionPreference = 'Continue'

function Detener([string]$patron, [string]$descripcion, [string]$rutaObligatoria = '') {
    $procesos = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match $patron -and
        (-not $rutaObligatoria -or $_.CommandLine -like "*$rutaObligatoria*")
    }
    foreach ($p in $procesos) {
        if ($p.ProcessId -eq $PID) { continue }
        Write-Host "  Deteniendo $descripcion (PID $($p.ProcessId))"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if (-not $procesos) { Write-Host "  ${descripcion}: no estaba corriendo" }
}

$raiz = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

Write-Host 'Deteniendo el stack JUCHLM ...' -ForegroundColor Cyan
if (-not $SoloOpenWA) {
    Detener 'whatsapp-juchlm\\bridge\.py' 'puente WhatsApp' $raiz
}
if (-not $SoloPuente) {
    Detener 'integrations\\OpenWA' 'OpenWA (gateway WhatsApp)' $raiz
}
if (-not $SoloPuente -and -not $SoloOpenWA) {
    Detener 'app\.py' 'portal ChatJUCHLM' $raiz
}
Write-Host 'Listo.' -ForegroundColor Cyan
Write-Host 'Para volver a levantar: .\integrations\whatsapp-juchlm\start_stack.ps1' -ForegroundColor DarkGray
