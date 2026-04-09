#!/usr/bin/env bash
# Evaluate agent.py on tau3-bench banking_knowledge (97 tasks).
# Prints pass@1 summary at the end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TAU3_DIR="$TASK_DIR/tau3-bench"
PYTHON="$TAU3_DIR/.venv/bin/python3"

# ── Settings ─────────────────────────────────────────────────────────────
AGENT_LLM="openai/gpt-5.4-mini"
USER_LLM="openai/gpt-4.1"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
MAX_STEPS=200
SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"

# ── Validate prerequisites ───────────────────────────────────────────────
if [ ! -d "$TAU3_DIR" ]; then
    echo "ERROR: tau3-bench not found. Run 'bash prepare.sh' first."
    exit 1
fi

if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

# ── Read retrieval config from agent.py ──────────────────────────────────
RETRIEVAL_VARIANT=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$TASK_DIR')
from agent import RETRIEVAL_VARIANT
# Block golden_retrieval — it provides oracle documents and is not a real strategy
if RETRIEVAL_VARIANT == 'golden_retrieval':
    print('ERROR: golden_retrieval is not allowed. Use bm25, openai_embeddings, or another real retrieval strategy.', file=sys.stderr)
    sys.exit(1)
print(RETRIEVAL_VARIANT)
" 2>&1) || { echo "ERROR: failed to read RETRIEVAL_VARIANT from agent.py"; exit 1; }

RETRIEVAL_KWARGS_JSON=$("$PYTHON" -c "
import sys, json
sys.path.insert(0, '$TASK_DIR')
from agent import RETRIEVAL_KWARGS
print(json.dumps(RETRIEVAL_KWARGS) if RETRIEVAL_KWARGS else '')
" 2>/dev/null || echo "")

echo "=== tau3-bench eval ==="
echo "Agent LLM:         $AGENT_LLM"
echo "User LLM:          $USER_LLM"
echo "Retrieval:         $RETRIEVAL_VARIANT"
echo "Concurrency:       $MAX_CONCURRENCY"
echo "Sample fraction:   $SAMPLE_FRAC"
echo ""

# ── Run evaluation ───────────────────────────────────────────────────────
cd "$TAU3_DIR"

if [ -n "$RETRIEVAL_KWARGS_JSON" ]; then
    RETRIEVAL_KWARGS_PY="$RETRIEVAL_KWARGS_JSON"
else
    RETRIEVAL_KWARGS_PY="None"
fi

"$PYTHON" -c "
import sys, json, os, random, math

sys.path.insert(0, '$TASK_DIR')

from agent import create_agent
from tau2.registry import registry
from tau2.runner import get_tasks, run_domain
from tau2.data_model.simulation import TextRunConfig

# Register the custom agent
registry.register_agent_factory(create_agent, 'hive_agent')

# Load all banking_knowledge tasks (97 total, no train/test split)
all_tasks = get_tasks('banking_knowledge')
all_ids = [t.id for t in all_tasks]

# Sample if requested
frac = float('$SAMPLE_FRAC')
n = max(1, math.ceil(len(all_ids) * frac))
if frac < 1.0:
    random.seed(42)
    task_ids = sorted(random.sample(all_ids, n))
else:
    task_ids = all_ids

print(f'Running {len(task_ids)} / {len(all_ids)} tasks', file=sys.stderr)

retrieval_kwargs_raw = $RETRIEVAL_KWARGS_PY
config = TextRunConfig(
    domain='banking_knowledge',
    agent='hive_agent',
    llm_agent='$AGENT_LLM',
    llm_user='$USER_LLM',
    num_trials=1,
    num_tasks=len(task_ids),
    task_ids=task_ids,
    max_steps=$MAX_STEPS,
    max_concurrency=int('$MAX_CONCURRENCY'),
    retrieval_config='$RETRIEVAL_VARIANT',
    retrieval_config_kwargs=retrieval_kwargs_raw if retrieval_kwargs_raw else None,
)

results = run_domain(config)
sims = results.simulations

passed = sum(1 for r in sims if r.reward_info and r.reward_info.reward == 1.0)
total = len(sims)
score = passed / total if total > 0 else 0.0
total_cost = sum((r.agent_cost or 0.0) + (r.user_cost or 0.0) for r in sims)

# Per-task breakdown
print('', file=sys.stderr)
print('=== Per-task results ===', file=sys.stderr)
for r in sims:
    reward = r.reward_info.reward if r.reward_info else 0.0
    status = 'PASS' if reward == 1.0 else 'FAIL'
    print(f'  {r.task_id}: {status}', file=sys.stderr)

# Standard hive output format
print('---')
print(f'pass_at_1:        {score:.6f}')
print(f'correct:          {passed}')
print(f'total:            {total}')
print(f'cost_usd:         {total_cost:.2f}')
"
