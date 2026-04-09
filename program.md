# τ-Knowledge Banking Agent

Improve a banking customer service agent to maximize pass@1 on τ-Knowledge banking_knowledge benchmark.

## Setup

1. **Read the in-scope files**:
   - `agent.py` — the file you modify. The banking customer service agent.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `prepare.sh` — installs τ3-bench. Do not modify.
2. **Run prepare**: `bash prepare.sh` to install τ3-bench.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The benchmark

τ-Knowledge banking_knowledge tests an agent's ability to act as a fintech customer service representative. The agent converses with a simulated user (gpt-4.1), searches a knowledge base of ~700 documents covering banking products and policies, reasons over complex interdependent policies, and executes multi-step tool calls to resolve customer requests.

Total: **97 tasks**. Tasks include opening accounts, disputing transactions, handling referrals, credit card retention, and more. Success requires getting the final database state exactly right.

Key challenge: 51 of 65 tools are **discoverable** — they are documented only in the knowledge base and must be found via search before they can be used.

## Experimentation

Each experiment runs on all 97 tasks. You launch it as: `bash eval/eval.sh`.

Use `SAMPLE_FRAC=0.1 bash eval/eval.sh` to run on ~10 tasks for fast iteration. Use `SAMPLE_FRAC=1.0` (default) for full evaluation.

**What you CAN do:**
- Modify `agent.py` — this is the only file you edit. Everything is fair game: system prompt, message handling, tool-use strategy, reasoning patterns, retrieval strategy, error recovery.
- Change `RETRIEVAL_VARIANT` in `agent.py`. Available: `bm25` (default, offline keyword search), `openai_embeddings` (embedding-based), `qwen_embeddings` (via OpenRouter), `grep_only`, `full_kb` (entire KB in context), `no_knowledge`.
- Change `RETRIEVAL_KWARGS` (e.g. `{"top_k": 10}`).

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or τ3-bench source code.
- Change the agent LLM (fixed to `openai/gpt-5.4-mini`).
- Change the user simulator model (fixed to `openai/gpt-4.1`).
- Use `golden_retrieval` (oracle documents — blocked by eval).
- Hardcode answers to specific task IDs.

**The goal: maximize pass@1.** A task "passes" when the agent achieves reward = 1.0 (final database state matches gold standard). pass@1 = fraction of the 97 tasks that pass.

**Simplicity criterion**: All else being equal, simpler is better. Verbose prompts hurt small models — be specific and concise.

**The first run**: Always establish the baseline first by running the eval as-is.

## Understanding the agent environment

The `domain_policy` string passed to your agent is NOT empty — it is a fully assembled prompt built from template files. For the `bm25` variant, it includes:

1. **Policy header**: Rho-Bank customer service guidelines — don't make up policies, use `get_current_time()`, transfer to human only as last resort, don't leak internal info.
2. **Retrieval instruction**: "Search the knowledge base using the provided `KB_search` tool."
3. **Additional instructions**: Full discoverable tool workflows (user tools and agent tools), authentication protocol (verify 2 of 4: DOB, email, phone, address), verification logging.

Your agent already receives instructions about authentication, discoverable tools, and KB search via `domain_policy`. Adding redundant instructions in your system prompt wrapper can hurt performance. Focus on restructuring HOW the information is presented, not duplicating WHAT is already there.

## Output format

The eval prints a summary:

```
---
pass_at_1:        0.0800
correct:          8
total:            97
cost_usd:         1.23
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	pass_at_1	cost_usd	status	description
```

1. git commit hash (short, 7 chars)
2. pass_at_1 (e.g. 0.080000) — use 0.000000 for crashes
3. cost in USD — use 0.00 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

## The experiment loop

LOOP FOREVER:

1. **THINK** — review results.tsv, form a hypothesis.
2. Modify `agent.py` with your experiment.
3. git commit
4. Run: `bash eval/eval.sh > run.log 2>&1`
5. Read results: `grep "^pass_at_1:" run.log`
6. If empty, check `tail -n 50 run.log` for errors.
7. Record in results.tsv (do not commit results.tsv).
8. If pass_at_1 improved, keep. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 60 minutes, kill it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous.
