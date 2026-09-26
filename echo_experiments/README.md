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
| `eval_full_game.py` | Runs a checkpoint (or the base model) through complete multi-turn episodes on the held-out benchmark set |
| `baseline_sanity_check.py` | Same full-game rollout, driven by an API builder (e.g. gpt-4o-mini) instead of a local checkpoint -- `--dataset benchmark` makes it directly comparable to `eval_full_game.py` |

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

The within-turn eval only checks a single frozen decision point. This
instead runs each condition through complete episodes (up to `max_turns`
turns, frozen directors, real game state) on the shipped 20-structure
benchmark (`data/structures_dataset_20.json` -- never used for training),
and reports `final_progress_mean` (does the structure actually get built,
not just per-turn `progress_delta`), `completed_rate`, `oracle_match_rate`,
`clarify_rate`, `invalid_move_rate`.

Both scripts default to `oracle_n=20`, `max_turns=20`, `gpt-4.1-mini`
directors -- keep these consistent across every condition below rather than
matching whatever a given training run used, since it's cross-condition
consistency (not matching train-time settings) that keeps the comparison
fair.

```bash
# trained checkpoints -- point --checkpoint at craft_echo_runs/<run_name>/checkpoint-<step>
python eval_full_game.py --checkpoint craft_echo_runs/<echo_run>/checkpoint-350 --label echo_step350 --report_to wandb
python eval_full_game.py --checkpoint craft_echo_runs/<rloo_run>/checkpoint-350 --label rloo_step350 --report_to wandb
python eval_full_game.py --checkpoint craft_echo_runs/<episode_return_run>/checkpoint-200 --label episode_return_step200 --report_to wandb

# untrained base model -- zero-shot reference point
python eval_full_game.py --label base --report_to wandb

# API builder, on the same held-out set for a direct comparison
python baseline_sanity_check.py --dataset benchmark --n_structures 20 \
  --builder_model gpt-4o-mini --director_mode api --director_model gpt-4.1-mini \
  --report_to wandb --run_name api_baseline_benchmark
```

`eval_full_game.py`'s checkpoint runs need a GPU (loads the 7B base model +
LoRA adapter); the API-builder run needs none, since both builder and
directors are OpenAI calls.
