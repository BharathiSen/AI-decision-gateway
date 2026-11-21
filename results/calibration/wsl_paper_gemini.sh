#!/bin/bash
set -euo pipefail
export SUDO_ASKPASS=/tmp/verifix_askpass.sh
cd /home/bharathi/AI-decision-gateway

# Load repo .env into this process (does not override existing exports).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

export LLM_PROVIDER="${LLM_PROVIDER:-gemini}"
export LLM_MODEL="${LLM_MODEL:-gemini-3.8-flash}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-none}"
export LLM_FALLBACK_PROVIDER="${LLM_FALLBACK_PROVIDER:-groq}"
export LLM_FALLBACK_MODEL="${LLM_FALLBACK_MODEL:-openai/gpt-oss-20b}"

mkdir -p results/pilot results/paper results/calibration
LOG=results/calibration/paper_v1_gemini_run.log
exec > >(tee -a "$LOG") 2>&1

if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
  echo "need GEMINI_API_KEY and/or GROQ_API_KEY (put them in .env)" >&2
  exit 2
fi

echo "=== paper_v1 start $(date -Is) ==="
echo "provider=$LLM_PROVIDER model=$LLM_MODEL effort=$LLM_REASONING_EFFORT"
echo "fallback=$LLM_FALLBACK_PROVIDER/$LLM_FALLBACK_MODEL"
echo "GEMINI_PRESENT=$(test -n "${GEMINI_API_KEY:-}" && echo yes || echo no)"
echo "GROQ_PRESENT=$(test -n "${GROQ_API_KEY:-}" && echo yes || echo no)"

PYTHONPATH=. python3 - <<'PY'
import os
from agent.decision_agent import load_dotenv, provider_config
from bench.outcomes import threshold_status
load_dotenv()
c = provider_config()
fb = (os.environ.get("LLM_FALLBACK_PROVIDER") or "").strip()
print("resolved", c["provider"], c["model"], "key", bool(c["api_key"]), "fallback", fb or None)
print("threshold_status", {n: threshold_status(n).get("status") for n in ("N0", "N1", "N2")})
PY

echo "=== mn -c ==="
sudo -A mn -c || true

echo "=== paper_v1 live 300 stratified ==="
sudo -A --preserve-env=GEMINI_API_KEY,GROQ_API_KEY,LLM_PROVIDER,LLM_MODEL,LLM_REASONING_EFFORT,LLM_FALLBACK_PROVIDER,LLM_FALLBACK_MODEL \
  env PYTHONPATH=. \
  LLM_PROVIDER="$LLM_PROVIDER" LLM_MODEL="$LLM_MODEL" LLM_REASONING_EFFORT="$LLM_REASONING_EFFORT" \
  LLM_FALLBACK_PROVIDER="$LLM_FALLBACK_PROVIDER" LLM_FALLBACK_MODEL="$LLM_FALLBACK_MODEL" \
  python3 -u scripts/finish_evaluation.py \
  --mode live --run-name paper_v1 --episodes 300 --paper-stratified

echo "PAPER_PIPELINE_DONE $(date -Is)"
if [ -t 0 ] && [ "${NONINTERACTIVE:-0}" != "1" ]; then
  read -r -p "Press Enter to close..."
fi
