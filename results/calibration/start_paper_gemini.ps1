# Visible launcher: Gemini paper_v1 on WSL
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "VeriFix paper_v1 Gemini launcher"
Set-Location "C:\Users\Admin\Projects\AI-decision-gateway"

Write-Host ""
Write-Host "=== VeriFix paper_v1 (Gemini, WSL, 300 stratified) ==="
Write-Host "Paste your Gemini API key below (hidden)."
Write-Host ""

$pass = $env:WSL_SUDO_PASS
if (-not $pass) {
  $secure = Read-Host "WSL sudo password" -AsSecureString
  $pass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
  if (-not $pass) { Write-Host "MISSING_SUDO"; Read-Host "Enter to close"; exit 2 }
  $env:WSL_SUDO_PASS = $pass
} else {
  Write-Host "WSL_SUDO_PASS already set."
}

if (-not $env:GEMINI_API_KEY) {
  $secure = Read-Host "GEMINI_API_KEY" -AsSecureString
  $gem = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
  if (-not $gem) { Write-Host "MISSING_GEMINI"; Read-Host "Enter to close"; exit 2 }
  $env:GEMINI_API_KEY = $gem.Trim().Trim("'").Trim('"')
}

$env:LLM_PROVIDER = "gemini"
if (-not $env:LLM_MODEL) { $env:LLM_MODEL = "gemini-3.8-flash" }
Write-Host "GEMINI len=$($env:GEMINI_API_KEY.Length) model=$($env:LLM_MODEL)"

"ready len=$($env:GEMINI_API_KEY.Length) model=$($env:LLM_MODEL) $(Get-Date -Format o)" |
  Set-Content -Path "results\calibration\_gemini_ready.txt" -Encoding ascii

# Wipe partial Groq paper_v1
@(
  "results\pilot\paper_v1.json",
  "results\pilot\paper_v1.state.json",
  "results\pilot\paper_v1_manifest.json",
  "results\pilot\paper_v1_halt.json",
  "results\pilot\paper_v1_experiments.json"
) | ForEach-Object { if (Test-Path $_) { Remove-Item -Force $_ } }
if (Test-Path "results\pilot\paper_v1_episodes") { Remove-Item -Recurse -Force "results\pilot\paper_v1_episodes" }
Write-Host "wiped partial paper_v1"

# Refresh freeze on host (same files as WSL via symlink)
$env:LLM_PROVIDER = "gemini"
python -c @"
from scripts.finish_evaluation import write_paper_freeze
import json, os
from pathlib import Path
m = write_paper_freeze(run_name='paper_v1', seed=1, episodes=300)
m['live_host'] = 'wsl:Ubuntu-22.04'
m['model_id'] = os.environ.get('LLM_MODEL') or 'gemini-3.8-flash'
m['note'] = 'Live host WSL Ubuntu-22.04; proposer Gemini. No further gate/baseline logic changes during the live paper batch.'
Path('results/paper/paper_v1_manifest.json').write_text(json.dumps(m, indent=2) + chr(10), encoding='utf-8')
print('freeze', m['model_id'], m['artifacts']['thresholds.yaml']['sha256'][:12])
"@

$pass | wsl -d Ubuntu-22.04 -- bash -c "cat > /tmp/.verifix_sudo_pass && chmod 600 /tmp/.verifix_sudo_pass"
wsl -d Ubuntu-22.04 -- bash -c "printf '%s\n' '#!/bin/sh' 'cat /tmp/.verifix_sudo_pass' > /tmp/verifix_askpass.sh && chmod 700 /tmp/verifix_askpass.sh && sed 's/\r$//' /home/bharathi/AI-decision-gateway/results/calibration/wsl_paper_gemini.sh > /tmp/wsl_paper_gemini.sh && chmod 700 /tmp/wsl_paper_gemini.sh && : > /home/bharathi/AI-decision-gateway/results/calibration/paper_v1_gemini_run.log && echo SETUP_OK"

$env:WSLENV = "GEMINI_API_KEY/u:LLM_PROVIDER/u:LLM_MODEL/u"
Start-Process -FilePath "wsl.exe" -ArgumentList @(
  "-d","Ubuntu-22.04","-e","bash","-lc",
  "export SUDO_ASKPASS=/tmp/verifix_askpass.sh NONINTERACTIVE=1; echo GEMINI_PRESENT=`$(test -n `"`$GEMINI_API_KEY`" && echo yes || echo no) PROVIDER=`$LLM_PROVIDER MODEL=`$LLM_MODEL; exec bash /tmp/wsl_paper_gemini.sh"
) -WindowStyle Normal

Write-Host "Opened WSL Gemini paper window. Leave it running."
Read-Host "Enter to close launcher"
