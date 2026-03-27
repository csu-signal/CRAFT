# CRAFT: Collaborative Reasoning Agents For Construction Tasks

> **Official repository** for the paper [CRAFT: Grounded Multi-Agent Coordination Under Partial Information](https://arxiv.org/pdf/2603.25268) (2026).

<p align="center">
  <img src="data/craft_director_views.png" alt="CRAFT Director Views" width="90%"/>
</p>

<p align="center">
  <img src="data/collab_reasoning_agents_craft.png" alt="CRAFT" width="90%"/>
</p>
 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

> **CRAFT** is a multi-agent benchmark for evaluating pragmatic communication 
> in large language models under strict partial information. Three director 
> agents with complementary but incomplete views of a 3D target structure must 
> coordinate through natural language to guide a builder agent toward the correct 
> configuration — a task no single agent can solve alone.

## Supported Models

**Frontier (API):** GPT-4o, GPT-4o-Mini, GPT-4.1-Mini, Claude-Sonnet-4.6, 
Gemini-2.5-Flash, Gemini-3-Flash, Gemini-3.1-Flash-Lite

**Open-weight (local):** Qwen-2.5 7B/14B/32B/72B, Llama-3-8B, Mistral-7B, 
Gemma-2-9B, DeepSeek-V2-Lite


## Overview

CRAFT evaluates a fundamental question: does stronger individual reasoning 
translate to better multi-agent coordination? Across 8 open-weight and 7 
frontier models, we find the answer is often **no** — smaller open-weight 
models frequently match or outperform frontier systems, and higher individual 
communication quality does not guarantee better collaboration.

The benchmark provides:
- A procedurally generated 3D block construction task with physics-constrained validation
- An oracle-assisted builder interface that isolates director communication as the performance bottleneck
- A suite of LLM judges that decompose failures into spatial grounding, mind modeling, and pragmatic sufficiency
```
Target 3D Structure
        │
        ▼
Structure Generator ──► 3 Private 2D Wall Projections (D1, D2, D3)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
               Director 1     Director 2     Director 3
               (left wall)    (far wall)     (right wall)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                              Public messages
                                    │
                                    ▼
                            Builder Agent
                      (oracle-verified candidates)
                                    │
                                    ▼
                          CRAFT Game Engine
                      (physics validation + logging)
```

## Task Setup

Three **Director** agents each receive a private 2D projection of a target 3D 
structure — one wall each. No director can see the full target. They must 
coordinate through natural language to instruct a **Builder** agent to place, 
remove, or request clarification on colored blocks on a 3×3 grid with up to 
3 vertical layers.

The Builder receives up to 5 oracle-verified candidate moves per turn and must 
identify which candidate the directors are describing — isolating director 
communication quality as the key variable.

**Information asymmetry operates at two levels:**
1. Each director sees only their wall of the target (partial target observability)
2. Each director's `<think>` block is private — only `<message>` is broadcast

## Repository Structure
```
CRAFT/
├── run_craft.py                  # Main experiment orchestrator (API models)
├── craft_run_open_weight.py      # Orchestrator for local open-weight models
├── game_state_tracking.py        # Core agents: DirectorAgent, BuilderAgent,
│                                 # EnhancedGameState, oracle interface
├── oracle.py                     # Oracle move enumeration and validation
├── structure_generator_v2.py     # Procedural 3D structure generation
├── environment.py                # Game environment and move execution
├── task_progress_tracker.py      # Per-turn progress metrics
├── builder_tools.py              # Builder move simulation tools
├── local_model_utils.py          # HuggingFace model loading utilities
├── judge_pragmatics.py           # PS judge implementation
├── sg_mm_judge_calls.py          # SG and MM judge implementation
├── structures_dataset_20.json    # 20 evaluation structures (7 simple,
│                                 # 8 medium, 5 complex; 21-25 blocks)
└── plotting_scripts/             # Analysis and visualization scripts
```

## Agents

### Director Agent (`game_state_tracking.py: DirectorAgent`)
Each director receives its private target wall view, the full current board 
state, and conversation history. It produces:
- `<think>`: unconstrained private spatial reasoning (not shared)
- `<message>`: a public instruction calibrated to what other agents know

Directors are assigned one of five personality archetypes — assertive, 
cautious, observant, skeptical, or synthesizer — deterministically via a 
seeded hash of `(structure_index, run, director_id)`, ensuring consistent 
role assignments across all model evaluations.

### Builder Agent (`game_state_tracking.py: BuilderAgent`)
The builder observes all director messages and up to 5 oracle-verified 
candidate moves per turn. It selects a move in structured format:
```
PLACE:block_code:position:layer:CONFIRM:reasoning
PLACE:block_code:position:layer:span_to:CONFIRM:reasoning  # large blocks
REMOVE:position:layer:CONFIRM:reasoning
CLARIFY:question
```

### Game Engine (`environment.py: EnhancedGameState`)
Validates moves against physical stacking constraints, updates board state, 
and logs per-turn progress metrics. Returns error messages on failure that 
feed back into the conversation history.

 
## Installation
```bash
git clone https://github.com/csu-signal/CRAFT
cd CRAFT
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY=...
export CLAUDE_API_KEY=...
export GEMINI_API_KEY=...
```

## Running Experiments

**API models:**
```python
# in run_craft.py, configure:
API_DIRECTOR_MODELS = ["gpt-4o", "claude-sonnet-4-6"]
BUILDER_MODEL       = "gpt-4o-mini"
DATASET_PATH        = "structures_dataset_20.json"
MAX_TURNS           = 20
RUN                 = 3
USE_ORACLE          = True
ORACLE_N            = 5

python run_craft.py
```

**Open-weight models:**
```python
# in craft_run_open_weight.py, configure LOCAL_MODELS paths
# then:
python craft_run_open_weight.py
```

Output is written to `{OUTPUT_DIR}/{director_model}_{builder_model}/craft_structure_{idx}_{run}.json`.

## Output Format

Each game log contains full trajectory data per turn:
```json
{
  "experiment_info": {...},
  "games": [{
    "structure_id": "structure_017",
    "complexity": "complex",
    "target_structure": {...},
    "target_spans": {...},
    "D1 Archetype": "cautious",
    "D2 Archetype": "synthesizer", 
    "D3 Archetype": "skeptical",
    "turns": [{
      "turn_number": 1,
      "structure_before": {...},
      "spans_before": {...},
      "director_responses": {
        "D1": {"internal_thinking": "...", "public_message": "..."},
        "D2": {"internal_thinking": "...", "public_message": "..."},
        "D3": {"internal_thinking": "...", "public_message": "..."}
      },
      "oracle_moves": [...],
      "move_attempted": {...},
      "move_executed": true,
      "builder_followed_oracle": true,
      "progress_data": {"overall_progress": 0.312, ...}
    }]
  }]
}
```

## LLM Judges

Three diagnostically independent judges evaluate director communication:

| Judge | Input | Evaluates |
|-------|-------|-----------|
| **Spatial Grounding (SG)** | `<think>` block only | Private reasoning quality — block ID, layer inference, executability |
| **Mind Modeling (MM)** | `<message>` only | Public message calibration — novelty, unique perspective, conflict resolution |
| **Pragmatic Sufficiency (PS)** | All director messages collectively | Whether collective output gave builder sufficient information to identify a correct move |

Run judges:
```bash
python sg_mm_judge_calls.py  # SG and MM
python judge_pragmatics.py   # PS
```

## Key Results

- Frontier models do not reliably outperform open-weight models: Mistral-7B 
  and Qwen-7B outperform 5 of 7 frontier systems
- Higher individual communication quality (SG, MM) negatively correlates 
  with task progress at the model level
- Frontier directors over-remove relative to oracle prescriptions 
  (remove gap up to +0.47), consuming the turn budget without advancing progress
- Conflict resolution (MM7) is universally near-zero (≤0.04) — no model 
  successfully models the joint listener in practice

## Citation
```bibtex
@misc{nath2026craft,
      title={CRAFT: Grounded Multi-Agent Coordination Under Partial Information}, 
      author={Abhijnan Nath and Hannah VanderHoeven and Nikhil Krishnaswamy},
      year={2026},
      eprint={2603.25268},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.25268}, 
}

```

## License

MIT
