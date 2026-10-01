# echo_experiments

Trains the CRAFT builder with turn-level RL credit (ECHO, plus the
`episode_return` / `rloo_per_turn` baselines) and evaluates it with the
within-turn experiment: does the trained policy pick the progress-maximizing
candidate at a single, frozen decision point? 

## Files

| File | Role |
|---|---|
| `reward.py` | Builder reward -- driven by CRAFT's own progress tracker (`overall_progress` delta) |
| `rollout.py` | One live episode: 3 frozen directors + 1 trainable builder per turn |
| `advantages.py` | `echo` / `episode_return` / `rloo_per_turn` advantage computation|
| `trainer.py` | `CRAFTEchoTrainer(GRPOTrainer)` -- shared by all three advantage modes |
| `data_split.py` | Training-only structure pool |
| `train.py` | CLI: trains one of the three conditions |
| `build_eval_pool.py` | Builds the frozen within-turn eval set (fixed director discussions, replayed identically to every checkpoint) |
| `eval_within_turn.py` | Scores one checkpoint against the frozen pool: best-candidate rate, regret, compliance rate, P(CLARIFY) |
| `eval_harness.py` | The shared full-game eval protocol: structures, per-episode seeds and starting boards, oracle setting, resume, result files |
| `eval_full_game.py` | Full-game eval of a local builder: a trained checkpoint or a zero-shot base model of any size (optionally 4-bit) |
| `baseline_sanity_check.py` | Full-game eval of an API builder (OpenAI, Gemini or Anthropic) under the same protocol |
| `api_builders.py` | Provider adapters for API builders (reasoning-model token budgets, temperature fallback, retries) |
| `eval_results.py` | Per-episode metrics and the per-run JSON / CSV result files |
| `analyze_evals.py` | CLI: figures and tables from a directory of eval results |
| `analysis/` | The analysis package: `registry.py` (conditions, metrics, colours), `stats.py`, `plots.py`, `tables.py` |

## Setup

Beyond `CRAFT/requirements.txt`, this needs the same extras `echo-edp` uses:
```bash
pip install torch transformers trl peft datasets python-dotenv
```
A `.env` (or exported env vars) with `OPENAI_API_KEY` is required

All commands below are run from inside `CRAFT/echo_experiments/`.

## 1. Build the training pool

```bash
python data_split.py --n 500
```
Writes `data/train_structures.json`, generated fresh via
`structure_generator_v2.generate_dataset` with a seed distinct from the
benchmark's, so training never touches `data/structures_dataset_20.json`.

## 2. Train the three conditions

```bash
python train.py --mode echo           --steps 300
python train.py --mode episode_return --steps 300
python train.py --mode rloo_per_turn  --steps 300
```
Each writes a LoRA checkpoint to `craft_echo_runs/<run_name>/final_model`.

To continue an interrupted run rather than starting over:
```bash
python train.py --mode echo --steps 350 --resume_from_checkpoint craft_echo_runs/<run_name>/checkpoint-175
```
Restores the LoRA weights, optimizer state, and `global_step` from that
checkpoint. Note this always starts a **new**, freshly-timestamped
`run_name`/wandb run (not conditioned on `--resume_from_checkpoint`) that
picks up logging from the restored step -- so in the wandb UI it shows up
as a second run object, not the original run's curve continuing on the same
page. View both runs together (e.g. select both in one panel) to see the
full before/after trend as one continuous curve.


## 3. Build the frozen within-turn eval pool (once)

```bash
python build_eval_pool.py --turns_per_structure 6
```
Uses the shipped 20-structure benchmark (`data/structures_dataset_20.json`)
and a real, frozen director+builder pass (builder here is only a *reference
policy* that advances the board between snapshots -- not one of the
checkpoints being compared) to capture `(board_state, director_discussion,
oracle_candidates)` tuples. Writes `data/within_turn_eval_pool.json`. Build
this once and reuse it for every checkpoint below -- that's what keeps the
comparison uncounfounded.

## 4. Evaluate each checkpoint against the same frozen pool

```bash
python eval_within_turn.py --base_model Qwen/Qwen2.5-1.5B-Instruct --label base
python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label echo
python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label episode_return
python eval_within_turn.py --checkpoint craft_echo_runs/.../final_model --label rloo_per_turn
```
Each run appends one row to `within_turn_results.csv`: `label,
best_candidate_rate, regret_mean, regret_std, compliance_rate,
clarify_rate, n_total` -- the reporting table from the within-turn
experiment design.

## 5. Full-game evaluation (held-out benchmark set)

Runs each condition through complete episodes (up to `--max_turns`, frozen
gpt-4.1-mini directors, real game state) on the shipped 20-structure
benchmark (`data/structures_dataset_20.json`, never used for training; the
harness checks it doesn't overlap the training pool). Both scripts share one
protocol (`eval_harness.py`): episode `k` of structure `s` gets the same
seed, starting board, director order and oracle candidates in every
condition, so conditions can be compared structure by structure. Each
episode is saved as it finishes; rerun with the same `--run_name` and
`--resume` to continue a crashed run.

Give every condition a unique `--label` and put them all in the same
`--out_dir`. Run from `echo_experiments/`, adding
`--episodes_per_structure 3 --out_dir eval_results_v2` to each command.
Model weights download to `$HF_HOME` (set to `/data/sifat/hf_cache` on the
`craft` conda env, since the root disk can't hold 32B/72B checkpoints):

```bash
# local builders (eval_full_game.py); --gpus picks the GPU, --quantize 4bit for large models
python eval_full_game.py --gpus 0 --label base_7b
python eval_full_game.py --gpus 0 --label echo --checkpoint /data/craft_echo_runs/<echo_run>/checkpoint-350
python eval_full_game.py --gpus 0 --label rloo --checkpoint /data/craft_echo_runs/<rloo_run>/checkpoint-350
python eval_full_game.py --gpus 0 --label grpo --checkpoint /data/craft_echo_runs/<episode_return_run>/checkpoint-350
python eval_full_game.py --gpus 0 --label base_14b --base_model Qwen/Qwen2.5-14B-Instruct
python eval_full_game.py --gpus 1 --label base_72b --base_model Qwen/Qwen2.5-72B-Instruct --quantize 4bit

# API builders (baseline_sanity_check.py); no GPU needed
python baseline_sanity_check.py --label api_gpt-4.1-mini      --builder_model gpt-4.1-mini
python baseline_sanity_check.py --label api_claude-sonnet-4-6 --builder_model claude-sonnet-4-6   # needs ANTHROPIC_API_KEY
```

Other protocol options: `--prompt_style cot` (reason first, answer on the
last line), `--no_oracle` (hide the candidate moves from the builder; use a
separate `--out_dir`, e.g. `eval_results_v2_no_oracle`), and
`--part_type empty` (start every episode from an empty board, as
`run_craft.py` does). Every run records its full settings in its JSON.

## 6. Analyze eval runs (figures + tables)

Every eval run (`eval_full_game.py` or `baseline_sanity_check.py`) writes
`<out_dir>/<run_name>.json` when it finishes. The analysis reads every JSON
in one results directory and turns them into paper figures and tables. It
needs no GPU and takes a few seconds, so rerun it whenever new runs land.

```bash
cd echo_experiments

# everything in eval_results_v2/, compared against the zero-shot Qwen2.5-7B run (label base_7b)
python analyze_evals.py --results eval_results_v2 --out analysis_out

# the no-oracle ablation is a separate experiment: analyze its own directory
python analyze_evals.py --results eval_results_v2_no_oracle --out analysis_out_no_oracle

# a subset of conditions, or only some metrics
python analyze_evals.py --results eval_results_v2 --labels base_7b grpo rloo echo
python analyze_evals.py --results eval_results_v2 --metrics final_progress completed invalid_move_rate
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--results` | `eval_results` | directory of per-run `*.json` files (`*.partial.jsonl` logs are ignored) |
| `--out` | `analysis_out` | where figures and tables are written (created if missing) |
| `--labels` | all | only these eval `--label`s |
| `--reference` | `base_7b` | label every condition is paired against in the comparison table |
| `--metrics` | all non-diagnostic | one bar chart per metric |
| `--table_metrics` | all non-diagnostic | table columns; name a diagnostic metric (`reward_mean`, `director_failure_rate`, `completion_tokens_mean`) to add it back |
| `--figures` | `bar progress_curve` | which figure types to draw |

Outputs in `--out`:

| File | Contents |
|---|---|
| `bar_<metric>.png` | one bar per condition (mean ± SEM), value above each bar, arrow on the y-axis for which direction is better, legend below; ECHO outlined in black |
| `progress_curve.png` | cumulative progress vs turn, one line per condition with a ± SEM band |
| `legend_bar.png`, `legend_line.png` | the legend alone, for assembling multi-panel figures |
| `results.md` | main table (mean ± SEM, best per column in bold) and paired differences vs `--reference` |
| `results.tex` | the main table as a booktabs LaTeX table (`\usepackage{booktabs}`) |
| `results.csv` | every estimate, SEM and paired test as raw numbers |

The analysis refuses to run when the runs in a directory don't belong
together, and warns when they're comparable but set up differently:

* **Error:** different structure files or oracle settings
  (`structures_sha`, `oracle_in_prompt`). Put each setting in its own
  directory.
* **Error:** one `--label` used for two different models. Labels must be
  unique per condition, e.g. `base_7b` / `base_14b`.
* **Warning:** runs differ in director model, `oracle_n`, `max_turns`,
  number of structures, episodes per structure or starting-board mode.

**Statistics.** Structures are the unit of analysis. Each structure's
episodes are averaged first, so 20 structures × 3 episodes gives n = 20, not
60. Tables and error bars show mean ± SEM (SD / √n over the structure
means), which is roughly a 68% interval, not 95%. Comparisons against
`--reference` are paired on the structures both conditions ran: Δ ± SEM of
the per-structure differences, with a sign-flip permutation test,
Holm-corrected across metrics (the † in the tables).

**Names, colours and order** come from the eval label and the run's model:
`echo`, `rloo`, `grpo`, `cot_*`, `base_*` (e.g. "Qwen2.5-72B 4-bit
(zero-shot)"), and `api_*` (named after `--builder_model`, e.g. "Claude
Sonnet 4.6"). Colours match the paper's existing figures where the same
model appears there. The progress curve needs the per-turn progress log,
which only runs made with the current harness contain.

**Extending it.** To add a method or baseline, add a `Method` in
`analysis/registry.py`; its key is matched as a prefix of the eval label.
To give an API or base model its own colour, add it to `MODEL_STYLES` in
the same file. For a new metric, compute it per episode in
`eval_results.episode_metrics`, then add a `Metric` to the registry. For a
new figure, add a function to `analysis/plots.py` and register it in
`PER_METRIC` (drawn once per metric) or `SINGLE` (drawn once).
