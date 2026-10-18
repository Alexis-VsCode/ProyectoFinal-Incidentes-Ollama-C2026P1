# Solo lectura. No instala, no cambia políticas ni modifica la configuración.
$ErrorActionPreference = 'Continue'
Write-Output 'Sistema operativo'
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
Write-Output 'CPU y memoria'
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object @{Name='RAM_GB';Expression={[math]::Round($_.TotalPhysicalMemory / 1GB, 1)}}
Write-Output 'Tarjetas gráficas'
Get-CimInstance Win32_VideoController | Select-Object Name
Write-Output 'Espacio libre'
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name='Libre_GB';Expression={[math]::Round($_.Free / 1GB, 1)}}
foreach ($programa in @('py','python','ollama','git','code')) {
    if (Get-Command $programa -ErrorAction SilentlyContinue) {
        Write-Output "Disponible: $programa"
        & $programa --version
    } else {
        Write-Output "No detectado en PATH: $programa"
    }
}
