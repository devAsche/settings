$env:USERDOMAIN = $env:COMPUTERNAME
$action = New-ScheduledTaskAction -Execute 'C:\AI model\venv\Scripts\python.exe' -Argument 'main.py --listen 127.0.0.1 --port 8188' -WorkingDirectory 'C:\AI model\ComfyUI'
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 72) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'MuseLaunchComfyUI' -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'MuseLaunchComfyUI'
'launched'
