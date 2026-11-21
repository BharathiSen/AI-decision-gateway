# Visible launcher for paper_v1 live batch on WSL.
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "VeriFix paper_v1 launcher"

Write-Host ""
Write-Host "=== VeriFix paper_v1 (WSL live, 300 stratified) ==="
Write-Host "Stage A passed. Freeze rewritten. This step needs GROQ_API_KEY."
Write-Host ""

$pass = $env:WSL_SUDO_PASS
if (-not $pass) {
  $secure = Read-Host "WSL sudo password" -AsSecureString
  $pass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
  if (-not $pass) { Write-Host "MISSING_SUDO"; Read-Host "Enter to close"; exit 2 }
  $env:WSL_SUDO_PASS = $pass
} else {
  Write-Host "WSL_SUDO_PASS already set in this session."
}

$groq = $env:GROQ_API_KEY
if (-not $groq) {
  $secure = Read-Host "Paste GROQ_API_KEY (input hidden)" -AsSecureString
  $groq = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
  if (-not $groq) { Write-Host "MISSING_GROQ"; Read-Host "Enter to close"; exit 2 }
  $env:GROQ_API_KEY = $groq
} else {
  Write-Host "GROQ_API_KEY already set in this session (len=$($groq.Length))."
}

Write-Host "Installing askpass and opening WSL paper batch window..."

$pass | wsl -d Ubuntu-22.04 -- bash -c "cat > /tmp/.verifix_sudo_pass && chmod 600 /tmp/.verifix_sudo_pass"
if ($LASTEXITCODE -ne 0) { Write-Host "askpass pass write failed"; Read-Host "Enter to close"; exit 2 }

wsl -d Ubuntu-22.04 -- bash -c "printf '%s\n' '#!/bin/sh' 'cat /tmp/.verifix_sudo_pass' > /tmp/verifix_askpass.sh && chmod 700 /tmp/verifix_askpass.sh && sed 's/\r$//' /home/bharathi/AI-decision-gateway/results/calibration/wsl_paper_pipeline.sh > /tmp/wsl_paper_pipeline.sh && chmod 700 /tmp/wsl_paper_pipeline.sh && echo SETUP_OK"
if ($LASTEXITCODE -ne 0) { Write-Host "pipeline setup failed"; Read-Host "Enter to close"; exit 2 }

$env:WSLENV = if ($env:WSLENV) { "$($env:WSLENV):GROQ_API_KEY/u" } else { "GROQ_API_KEY/u" }

Start-Process -FilePath "wsl.exe" -ArgumentList @(
  "-d", "Ubuntu-22.04",
  "-e", "bash", "-lc",
  "export SUDO_ASKPASS=/tmp/verifix_askpass.sh; echo GROQ_PRESENT=`$(test -n `"`$GROQ_API_KEY`" && echo yes || echo no); exec bash /tmp/wsl_paper_pipeline.sh"
) -WindowStyle Normal

Write-Host "Opened WSL paper_v1 window. Leave it running (~hours)."
Write-Host "This launcher window can stay open to keep the key in session env."
Read-Host "Enter to close launcher"
