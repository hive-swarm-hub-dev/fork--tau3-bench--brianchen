#!/usr/bin/env bash
# Set up tau3-bench. Run once before eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU3_DIR="$SCRIPT_DIR/tau3-bench"

echo "=== tau3-bench: prepare ==="

# ── Clone tau3-bench (pinned to main) ──────────────────────────────────────
if [ -d "$TAU3_DIR" ]; then
    echo "tau3-bench already cloned at $TAU3_DIR"
else
    echo "Cloning tau3-bench..."
    git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "$TAU3_DIR"
fi

# ── Install dependencies ──────────────────────────────────────────────────
cd "$TAU3_DIR"

echo "Installing tau3-bench with knowledge extras..."

if command -v uv &>/dev/null; then
    uv sync --extra knowledge 2>&1
else
    pip install -e ".[knowledge]" 2>&1
fi

# ── Verify setup ──────────────────────────────────────────────────────────
VENV_PYTHON="$TAU3_DIR/.venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

echo ""
echo "Verifying setup..."
"$VENV_PYTHON" -c "
from tau2.runner import get_tasks
tasks = get_tasks('banking_knowledge')
print(f'  banking_knowledge domain: {len(tasks)} tasks loaded')
" 2>&1

echo ""
echo "=== Preparation complete ==="
echo "Make sure OPENAI_API_KEY is set before running eval."
