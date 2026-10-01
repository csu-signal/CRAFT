# CRAFT without information separation (single full-information Builder)

One Builder receives all three wall views (D1 left, D2 far, D3 right) and must build a structure consistent with every view. The rest of CRAFT is kept: empty board, 20-turn budget, one move per turn, same metrics. No Directors and no dialogue.

## Files

| File | Role |
|---|---|
| `craft_full_info_env.py` | Engine with no dependencies: physics, view projection, metrics, oracle, scripted builders |
| `run_full_info_builder.py` | Prompt, LLM backends, game loop, CLI, summary |
| `craft_directors.py` | Director mode (`--directors`): CRAFT Director and Builder prompts, Director loop |
| `craft_sim_tool.py` | Builder move-simulation tool (`--sim-tool`): `simulate_move` port, tool calling, budget loop |
| `test_full_info.py` | Self-tests; no model or network needed |

Everything is standard-library Python 3.9+. All HTTP calls use `urllib`, so no `openai`, `anthropic` or `datasets` packages are needed. The files do not import the CRAFT repo, so they run anywhere.

## Quick start

```bash
# 0. self-tests (~10 s)
python test_full_info.py structures_dataset_20.json

# 1. harness check + information ceiling, no model
python run_full_info_builder.py --structures structures_dataset_20.json \
    --out-dir runs/ceiling_views --backend scripted-views

# 2. local test drive
ollama pull qwen2.5:7b-instruct
ollama serve            # no OLLAMA_CONTEXT_LENGTH needed: num_ctx is sent per request
python run_full_info_builder.py --structures structures_dataset_20.json \
    --out-dir runs/qwen7b_smoke --backend ollama --model qwen2.5:7b-instruct --limit 2 --verbose

# 3. full local run (paper protocol: 20 structures x 3 runs)
python run_full_info_builder.py --structures structures_dataset_20.json \
    --out-dir runs/qwen7b --backend ollama --model qwen2.5:7b-instruct --runs 3

# 4. API runs
OPENAI_API_KEY=...    python run_full_info_builder.py ... --backend openai    --model gpt-4o-mini
GEMINI_API_KEY=...    python run_full_info_builder.py ... --backend gemini    --model gemini-3-flash-preview
ANTHROPIC_API_KEY=... python run_full_info_builder.py ... --backend anthropic --model <claude model id>
# any OpenAI-compatible server (vLLM, llama.cpp):  --backend openai --base-url http://host:8000/v1
```

Game files are written one at a time. Re-running the same command skips finished games, so a crashed API run resumes where it stopped. An `--out-dir` is locked to its config: if a later run uses different settings, the script exits instead of mixing results. Use a new directory or `--overwrite`.

## What changed relative to CRAFT, and why

| Aspect | CRAFT | Here | Reason |
|---|---|---|---|
| Information | 3 Directors, each with 1 view; Builder sees messages only | Builder sees all 3 target views directly | The condition requested |
| View format | Director JSON (`row_k` = layer, left-to-right per seat) | Same JSON and the same interpretation notes. `--view-format annotated` adds coordinates | Holds the representation constant; only the separation is removed |
| CLARIFY | Allowed | Removed; `DONE` added | Nobody is there to answer a clarification. `DONE` freezes the board so a finished build is not damaged by forced moves. `--no-done` disables it |
| Feedback | Builder sees only the current turn; Directors relay failures | Builder sees its own move log with engine errors (`--history-window`) | Without Directors, a failed move would otherwise repeat deterministically and burn turns |
| Oracle candidates | Shown (N=5) in the paper's main runs | Off by default; `--oracle-in-prompt` turns them on | The oracle comes from the true 3D target, so it leaks interior information the views cannot provide. It is always computed for logging |
| Physics | CRAFT engine | Strict: layer must equal stack height, height cap 3, dominoes need an adjacent equal-height partner, removals need the exact domino partner | Matches the engine behaviour in the paper (layer/span errors). Note that `run_single_builder._place_block` is more permissive (see below) |
| Turn budget | 20 | 20 (`--turns`) | Unchanged |
| Builder decoding | temperature 0.1, 250 tokens, first line | temperature 0.1, 400 tokens, robust line extraction, 2 free format retries | Same as `run_single_builder_local.py`. `--reasoning` allows a chain of thought with the move on the last line |

## Director mode (`--directors`)

Plays the full CRAFT game instead of the single Builder. Each turn, a random 1–3 unique Directors speak in random order. Each Director sees the board and the whole dialogue so far, including the Builder's reports of executed or failed moves and its clarification questions. The Builder then sees only this turn's Director messages and the board, and makes one `PLACE` / `REMOVE` / `CLARIFY` move. The Builder never sees any view, as in CRAFT.

| Flag | Default | Meaning |
|---|---|---|
| `--directors` | off | Enable Director mode. Without it the script behaves exactly as before |
| `--director-views all\|own` | `all` | `all`: every Director gets all three target views (no information separation). `own`: each Director gets only its wall (standard CRAFT, the control) |
| `--director-backend`, `--director-model`, `--director-base-url`, `--director-api-key-env` | same as Builder | Directors and Builder can use different models (CRAFT: varied Directors, fixed GPT-4o-mini Builder) |
| `--director-temperature` | 0.7 | CRAFT Director setting |
| `--director-max-tokens` | 512 | Paper: 512 open-weight, 2000 GPT, 3000 Claude/Gemini |

Builder settings (`--backend`, `--model`, `--temperature`, `--max-tokens`, `--oracle-in-prompt`, `--reasoning`) apply to the Builder in both modes. The paper's main results gave the Builder oracle candidates, so add `--oracle-in-prompt` when comparing against Table 1. In Director mode the oracle goes to a Builder that sees no views, exactly as in CRAFT. `DONE` is disabled, because CRAFT has none.

Prompts are ported from `director_agent.py` and `builder_agent.py`. In `own` mode they are verbatim apart from whitespace. In `all` mode only the parts that assume a single private view change:

- the target-view section becomes all three views;
- "describe what only you can see" becomes "avoid repeating another Director's instruction";
- one rule is added to both prompts: a Director describing a cell on another wall names that wall and uses its owner's frame ("on D3's wall, the bottom left").

The Builder prompt keeps CRAFT's line saying dominoes cannot span (1,1)/(2,1), so that both conditions stay comparable with the original. Remove it in `craft_builder_prompt` to test its effect. When `structure_generator_v2` is importable (running inside the CRAFT repo), its block and coordinate reference strings are used; otherwise stand-ins are used. `reference_strings` in each game file records which.

Director-mode game files add `dialogue`, and per turn `speakers`, `directors` (message, private thinking, raw text, silent flag, usage), `discussion` and `director_usage`. The summary adds `director_stats` and `tokens_directors`. File names are tagged `dirs-<views>_<director model>+<builder model>`.

## Builder assistance: oracle candidates or the simulation tool

Builder assistance has three settings, and the two flags are in an argparse mutually exclusive group, so passing both is an error:

- **Neither (default):** no assistance.
- **`--oracle-in-prompt`:** up to `--oracle-n` verified progress moves are listed in the prompt.
- **`--sim-tool`:** the Builder gets CRAFT's `simulate_move` tool and can dry-run up to `--max-simulations` moves per turn (default 3). Simulations do not use game turns.

Both settings work in single-Builder and Director mode.

Ported from the CRAFT repo:
- `simulate_move` (`agents/builder_tools.py`), keeping the same return keys and hint wording;
- the tool schema and budget logic from `generate_move_with_tools`, including the forced final answer when a round would exceed the budget or the budget runs out;
- the tool-mode system message, and the tool-mode prompt addendum verbatim in Director mode. Single-Builder mode uses the same rules minus the Director-specific ones.

Tool calling is native for Ollama, OpenAI-compatible endpoints (including Gemini) and Anthropic. Qwen-style `<tool_call>{...}</tool_call>` text is also accepted as a call.

`--sim-score` controls what the tool reports:

| | `target` (default, as CRAFT) | `views` |
|---|---|---|
| `overall_progress` | Progress toward the true target, so it leaks target information (interior cells included), like the oracle | `view_match` |
| `structurePlacement` | The move is one of the oracle's verified progress moves | The move makes the affected wall cells match the views |

`sidePlacement` is the wall-view check in both settings. CRAFT's `EnhancedGameState` is not available here, so these two correctness flags follow the stated definitions rather than CRAFT's internal code. `views` is the setting that gives the full-information single Builder feedback without revealing anything beyond the views.

`--sim-require N` (default 0) makes the tool mandatory: the Builder must call `simulate_move` at least N times before its final answer. Default 0 is CRAFT's behaviour, where the tool is optional. Small local models such as Qwen2.5-7B often ignore optional tools, and `[SIM summary] 0 simulation call(s)` appears every turn. Under `--sim-require`:
- APIs that support it get `tool_choice` forced (`required` for OpenAI-compatible endpoints, `any` for Anthropic). If an endpoint rejects `required`, the request falls back to `auto`.
- Ollama has no `tool_choice`, so a model that answers without calling the tool is re-prompted ("You MUST call simulate_move..."), at most 2 times per turn. If it still does not call the tool, the turn proceeds and is logged with `sim_required_unmet`.
- Each re-prompt is an extra model call and counts toward the Builder token totals. It also departs from CRAFT's optional-tool design, so report it as a deviation.

**Diagnosing a model that does not call the tool.** Run the probe, which tests tool calling outside the game:
```bash
python craft_sim_tool.py --backend ollama --model qwen2.5:14b-instruct
```
It reports how many of 3 trials returned a structured call, a call written as text, or no call. For Ollama, `ollama show <model>` should list `tools` under Capabilities. When the model skips the tool, the console prints the start of its reply ("model said: ...") for each re-prompt. In tool mode the system prompt no longer carries the "output EXACTLY ONE line, no JSON" addendum, which contradicted calling a tool. Calls written as text are recovered (Qwen `<tool_call>` tags, JSON with `name`/`arguments`, bare `{"move": ...}`, `simulate_move(...)` Python style, `SIMULATE:PLACE:...` lines).

**Empty replies and `--sim-tool-mode text`.** With some Ollama models the native tool-calling path returns completely empty replies on some turns: the model generated tokens, but the server's own tool-call parsing dropped them. The harness now:
- re-sends an empty reply up to 2 times with a new seed before treating the turn as a miss, and prints `[SIM empty reply] completion_tokens=N done_reason=...`. A large `completion_tokens` with empty text means the reply was swallowed. A value near 0 means the model emitted nothing;
- offers `--sim-tool-mode text` (default `native`): the tool is described in the system prompt, the model writes `<tool_call>{...}</tool_call>`, results come back in `<tool_response>` tags, and the harness parses the calls itself. This bypasses the server's tool parsing and works for models without tool support. The probe accepts `--tool-mode text` as well.

Text mode departs from CRAFT's native-tool design, so report it. Persistent empty replies are expensive: each turn can use up to about 9 model calls before the harness gives up.

Each turn logs `simulations` (move and result), `sim_calls`, `sim_forced_final`, `final_was_simulated_ok`, `final_failed_in_sim`, `sim_nudges` and `sim_required_unmet`. The summary adds `sim_stats`. Every tool round is a model call, so simulation tokens appear in the Builder token totals.

## Metrics (all recorded after every turn)

- **`progress` (OP), `completion` (CP), `iou`, `position_accuracy` (PA)** are the paper's App. B.5 definitions over all 9 cells, so they compare directly with CRAFT Table 1. `iou` is checked against CRAFT's `calculate_iou_board` in the tests.
- **`visible_*`** are the same metrics restricted to the 7 wall cells.
- **`view_match`** is the fraction of the 81 (director, layer, cell) view entries the built board reproduces. `views_exact` means all three views match. This is the task's own objective: "consistent with all three views".
- **`views_ceiling`** (stored in every game file) scores the build a perfect reasoner would make from the views alone, measured against the true target.
- **Per turn:** `oracle_adherent` (move is one of the verified progress moves toward the true target), `taxonomy` (paper App. C: correct / engine-layer / engine-span / wrong-position / wrong-color / wrong-span), `error_type`, raw model text, token usage.

### The views cannot determine the whole target

Cells (1,1) and (2,1) appear in no view. A domino with one half in the interior shows as a small block on its wall. So a builder that matches all three views exactly still does not reproduce the target. On the 20 structures, the scripted-views builder satisfies every view on every structure but reaches only:

| | OP | CP | IoU | PA | view_match |
|---|---|---|---|---|---|
| Views-only ceiling (mean of 20) | 0.807 | 0.859 | 0.829 | 0.733 | 1.000 |
| True-target oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

For this condition, report `view_match` / `views_exact` as the main outcome. Full-board OP is still worth reporting for comparison with CRAFT, but it should be read against the 0.807 ceiling, not 1.0. The oracle needs 14 to 20 moves per structure; structure_020 needs exactly 20, so any single failed move on it rules out completion.

## GPT and other API Builders

Switching roles needs only flags (`--backend openai --model gpt-5.4-mini`, plus `--director-*` for the Directors). Points specific to OpenAI-style reasoning models:

- **`--reasoning-effort none|minimal|low|medium|high|xhigh`** (and `--director-reasoning-effort`, which defaults to the Builder's value when both roles use the same backend) is sent as `reasoning_effort`. It is not sent if omitted. GPT-5.4 mini defaults to `none`.
- **Native tools and reasoning do not mix on Chat Completions for GPT-5.4 and later.** The API rejects function tools unless the effort is `none` (confirmed for gpt-5.4-mini by the API's own error; the documentation describes the same rule for the 5.6 family). The harness runs a one-call preflight and exits with a message naming the options: `--reasoning-effort none`, or `--sim-tool-mode text`, which sends no function tools and so works with any effort. `--skip-preflight` disables the check.
- **Recommended combination for a reasoning Builder:** `--sim-tool-mode text --reasoning-effort medium --max-tokens 4000 --director-reasoning-effort none`. The Directors already write a visible `<think>` block (CRAFT prompt), so hidden reasoning on top of it mostly just consumes their token budget.
- **Token caps:** hidden reasoning tokens count against `--max-tokens`. At any effort above `none`, raise the caps (the paper used 2,000 for GPT-series Directors) or replies come back empty with `finish_reason: length`. The empty-reply diagnostic names this case.
- **Automatic parameter adaptation:** `max_tokens` becomes `max_completion_tokens`, and `temperature`, `seed` and `reasoning_effort` are dropped if the model rejects them (a `[param]` line is printed).
- **Tool messages** are sent without the extra `name` key (kept for the Gemini backend).
- **Smoke test before a full run:** `python craft_sim_tool.py --backend openai --model gpt-5.4-mini --reasoning-effort none` checks tool calling; then `--limit 1 --turns 3` checks a short game.

### Reading the reasoning-token count

`reasoning_tokens` is `0` when the API reports no hidden reasoning, and `None` when the backend does not report it at all (Ollama, Anthropic). GPT-5.4 mini defaults to `reasoning_effort: none`, so `0` is the correct value unless `--reasoning-effort low|medium|high` is passed. Two other checks: `total_tokens - prompt_tokens - completion_tokens` is `0` per call (nothing hidden), and `finish_reason` shows whether a reply was cut off. The Directors' `<think>` blocks are ordinary visible output and are counted under `output_visible_tokens`, not as reasoning tokens.

**Truncated Director replies:** when a Director reply ends with `finish_reason`/`done_reason` = `length` and has no complete `<message>...</message>`, it is treated as silent (flag `truncated` in the log, count in `director_stats`) instead of forwarding its unfinished reasoning to the Builder, which is what the ported CRAFT parser would do. The default `--director-max-tokens 512` is too low for GPT-series Directors: the paper used 2,000.

## Server errors, partial logs and stopping

- **Generation aborts:** an Ollama `HTTP 500 ... token repeat limit reached` means the model fell into a repetition loop and the server stopped it. Re-sending the same request fails the same way, so the harness retries immediately with a different seed (up to 2 times) instead of waiting out the backoff.

- **Retries:** a failed request (HTTP 429/5xx or a network error) is retried with backoff of 2, 4, 8, 16 and 32 seconds, up to `--http-retries` attempts (default 6; `1` fails fast). Each retry line shows the server's own error text, for example `{"error": "llama runner process has terminated..."}`. That text is the diagnosis: on Ollama it points to a crashed model runner or memory pressure (check `ollama ps` and `~/.ollama/logs/server.log`).
- **Partial logs:** after every turn the game is saved as `<game>.partial.json` (`"partial": true`, turns so far, and the dialogue in Director mode). The file is deleted when the game finishes, so an interrupted or aborted game still leaves a readable log.
- **Stopping:** after `--max-backend-fail-turns` consecutive turns whose model calls all failed (default 3, `0` = never stop) the run exits with a message and keeps the partial log. Re-running the same command skips finished games and starts the interrupted game again from turn 1.
- **Two Ollama models:** a Builder and Directors on different Ollama models can force the server to swap models between calls. On machines with limited memory this can trigger runner crashes. The two roles using the same model avoids the swap.

## Output layout

```
runs/<name>/
  _config.json                      config the directory is locked to
  _summary.json                     aggregate: final metrics mean/SEM, views ceiling, by complexity,
                                    turnwise curves, action/error/taxonomy counts, adherence, tokens
  <model>_<structure>_run<r>.json   one game: target, views, ceiling, and per turn: move, raw text,
                                    execution, error, oracle candidates/adherence, board before/after, metrics
```

`turnwise_mean.progress` has the same shape as the turnwise outcome dictionaries already in the project, so curves can be overlaid directly.

## Useful ablations

- `--view-format annotated`: separates perspective-taking (reading seat-relative views) from planning.
- `--show-current-views`: also shows the current board projected onto each wall, so progress is a direct diff.
- `--oracle-in-prompt`: comparable to CRAFT's oracle-assisted Builder (but see the leakage note above).
- `--history-window 0`: removes the move log, like the CRAFT Builder's memory.
- `--reasoning`: chain of thought allowed (`--max-tokens` defaults to 3000).

## Issues found in the existing code and data

1. **`structures_dataset_20.json` view sizes.** `director_views` has `size: 1` in all 540 cells. Paper App. B.4 says a domino whose two cells are both on a wall shows as size 2. The colours match a recomputation exactly; only the sizes are missing. This harness recomputes views from `structure` + `spans` by default (`--views-source dataset` uses the stored ones). If the CRAFT runtime also takes views from this file rather than recomputing them, the Directors in the logged games never saw a size-2 block. That is worth checking in `structure_generator_v2.get_director_views`.
2. **A builder-prompt rule that does not hold for this data.** The CRAFT builder prompt says *"A large block that is visible to ANY of the directors CANNOT span EITHER (1,1) or (2,1)"*. In this dataset, 25 of 131 dominoes touch (1,1) or (2,1). The rule is left out of this harness's prompt. In the original setup it gives the Builder false guidance on about 19% of dominoes.
3. **Permissive physics in the counterfactual harness.** `run_single_builder._place_block` ignores the requested layer for small blocks, does not enforce the height cap, and checks only equal heights (not adjacency) for dominoes. So `move_executed=True` in `run_single_builder_local.py` can mark a move as successful that the CRAFT engine would reject as a layer error. That could inflate executed rates or IoU in the factual/counterfactual comparison. It affects both arms equally, but not necessarily by the same amount.
