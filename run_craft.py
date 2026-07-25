# API mode, all models, all structures
# python run_craft.py --mode api

# # API mode, single model, single structure
# python run_craft.py --mode api --director gpt-4o-mini --structures 0

# # Local mode, specific model, with oracle
# python run_craft.py --mode local --director qwen-7b --oracle --oracle_n 5
# python run_craft.py --mode api --director gpt-4o-mini --oracle --oracle_n 5 --structures 0,1,2 --turns 5
# python run_craft.py --mode api --director gpt-4o-mini --structures 0,1,2 --turns 5
# # Local mode, no tools, specific structures
# python run_craft.py --mode local --director qwen-7b --no_tools --structures 0,1,2

# # Custom output dir
# python run_craft.py --mode api --director gpt-4o --output my_results --turns 15# 

import os
import json
import copy
import random
from datetime import datetime
from tqdm import tqdm
from typing import Dict
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from agents.common_ground_agent import CommonGroundAgent
# from chair_agent import ChairAgent
from local_model_utils import load_local_director_model, load_local_director_pipeline
from structure_generator_v2 import get_director_views as get_director_views_fn
from agents.environment import (
    EnhancedGameState, get_oracle_moves
)
from agents.director_agent import DirectorAgent
from agents.builder_agent import BuilderAgent
from agents.oracle import enumerate_correct_actions
import inspect
import time
import sys

print("EnhancedGameState from:", inspect.getfile(EnhancedGameState))
print("get_director_views sig:", inspect.signature(EnhancedGameState.get_director_views))
 
print([m for m in dir(EnhancedGameState) if not m.startswith("_")])

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def view_cell(v):
    return (v["color"], v["size"])

def view_distance(curr, targ):
    mism, total = 0, 0
    for d in ["D1", "D2", "D3"]:
        for r in ["row_0", "row_1", "row_2"]:
            for j in range(3):
                total += 1
                if view_cell(curr[d][r][j]) != view_cell(targ[d][r][j]):
                    mism += 1
    return mism, total

def compare_views(label, a, b):
    if a == b:
        print(f"[OK] {label}: views match")
        return True
    print(f"[MISMATCH] {label}")
    for d in ["D1", "D2", "D3"]:
        if a[d] != b[d]:
            print(f"  Director {d} differs:")
            print("  stored   :", json.dumps(a[d]))
            print("  recomputed:", json.dumps(b[d]))
    return False

def chair_history_only_directors(conversation_history: list[str], max_chars=2000) -> str:
    lines = [ln for ln in conversation_history if ln.startswith(("D1:", "D2:", "D3:"))]
    s = "\n".join(lines)
    return s[-max_chars:]
# ─────────────────────────────────────────
# MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────

def run_craft_experiments(
    dataset_path="structures_dataset_v2.json",
    structure_index=0,
    director_model_name="gpt-4.1-mini",
    builder_model_name="gpt-4o",
    common_ground_model_name="gpt-4.1-mini",
    api_key="None",
    max_turns=15,
    output_dir="craft_results",
    use_common_ground=False, 
    run = 1,
    lastPartType = None,
    shared_model = None, 
    shared_tokenizer = None,
    builder_tool_use = True,
    use_oracle=False,
    num_oracle = None,
    max_tokens = None
):
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    col_names = ['Turn', 'Failed Move', 'Clarify', 'Structure Placement', 'Side Placement', 'Overall Structure State', 'Group Agreement', 'Transcription' ]
    correctness = pd.DataFrame(columns=col_names)

    # ── Load structure ────────────────────────────────────────
    with open(dataset_path, "r") as f:
        loaded_structures = json.load(f)

    sample = loaded_structures[structure_index]

    # Fix span keys: JSON saves them as strings, need ints
    target_spans = {int(k): v for k, v in sample['spans'].items()}

    print(f"\nLoaded: {sample['id']} (complexity={sample['complexity']})")
    print(f"Metadata: {sample['metadata']}")
    
    swapped_dir_views = {d: {row: items[::-1] for row, items in rows.items()} if d in ["D2", "D3"] else rows for d, rows in sample["director_views"].items()} 
    target_structures_list = [{
        'id'            : sample['id'],
        'complexity'    : sample['complexity'],
        'structure'     : sample['structure'],
        'spans'         : target_spans,
        'director_views': swapped_dir_views,
        
        'metadata'      : sample['metadata']
    }]

    # ── Init results ──────────────────────────────────────────
    all_results = {
    'experiment_info': {
        'timestamp'      : current_time,
        'dataset_path'   : dataset_path,
        'structure_index': structure_index,
        'max_turns'      : max_turns,
        'models'         : {
            'director': director_model_name,
            'builder' : builder_model_name
        },
        
        'use_oracle'     : use_oracle,
        'num_oracle'     : num_oracle,
        'builder_tool_use': builder_tool_use,
        'use_common_ground': use_common_ground,
        'run'            : run,
        'max_tokens'     : max_tokens,
        'partial_completion': lastPartType,
            },
        'games': []
    }
    #add director builder combo to name the outer dir
    model_combo_dir = f"{output_dir}/{director_model_name}_{builder_model_name}/"
    os.makedirs(model_combo_dir, exist_ok=True)

    md_path  = f"{model_combo_dir}/craft_{sample['id']}_{run}.md"
    json_path = f"{model_combo_dir}/craft_{sample['id']}_{run}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)   
    main_grice_correctness_filename = f"{model_combo_dir}/craft_{sample['id']}_correctness_{run}.csv"

    with open(md_path, 'w') as f:
        f.write(f"# CRAFT Results — {sample['structure']}\n\n")
   
    # ── Run each structure ────────────────────────────────────
    for idx, structure_data in enumerate(tqdm(target_structures_list)):

        print(f"\n===== Structure {idx+1}: {structure_data['id']} =====")
        if shared_model is None: 
            print("calling api dirs ")
 
            director_agents = {
                did: DirectorAgent(
                    director_id=did,
                    use_api=True,
                    api_key=api_key,
                    model_name=director_model_name,
                    structure_index=structure_index,    
                    run=run,      
                    max_tokens=max_tokens,                  
                )
                for did in ["D1", "D2", "D3"]
            }
        else:
            print("calling local dirs ")
            director_agents = {
            did: DirectorAgent(
                director_id=did,
                use_api=False,
                model_name=director_model_name,
                local_model=shared_model,
                local_tokenizer=shared_tokenizer,
                structure_index=structure_index,  
                run=run,                       
            )
            for did in ["D1", "D2", "D3"]
        }


        builder_agent = BuilderAgent(api_key=api_key, model_name=builder_model_name)
        
        common_ground_agent = None
        if use_common_ground:
            common_ground_agent = CommonGroundAgent(
                use_api=True,
                api_key=api_key,
                model_name=common_ground_model_name
            )
        try:
            game_result, correctness, lastPartType = run_single_game(
                structure_data=structure_data,
                director_agents=director_agents, 
                builder_agent=builder_agent,
                # chair_agent=chair_agent,             
                common_ground_agent=common_ground_agent,
              
                max_turns=max_turns,
                structure_idx=idx,
                correctness=correctness,
                use_common_ground=use_common_ground, 
                run = run,
                lastPartType = lastPartType,
                use_tools = builder_tool_use,
                use_oracle=use_oracle,
                num_oracle = num_oracle
                
            )
        except Exception as e:
            print(f"Error in structure {idx}: {e}")
            game_result = {
                'structure_id': structure_data['id'],
                'error': str(e),
                'completed': False,
                'turns_taken': 0,
                'final_progress': 0.0
            }

        all_results['games'].append(game_result)

        with open(md_path, 'a') as f:
            f.write(f"\n## {structure_data['id']}\n")
            f.write(f"- Turns: {game_result['turns_taken']}\n")
            f.write(f"- Progress: {game_result.get('final_progress', 0):.3f}\n")
            f.write(f"- Completed: {game_result.get('completed', False)}\n")

        correctness.index.name = 'Transcript'
        print(correctness)
        correctness.to_csv(main_grice_correctness_filename)

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Results: {json_path}")
    return all_results, lastPartType


# ─────────────────────────────────────────
# SINGLE GAME
# ─────────────────────────────────────────

def run_single_game(
    structure_data,
    director_agents,
    builder_agent,
    # chair_agent,  
    common_ground_agent,
 
    correctness,
    max_turns=15,
    structure_idx=0,
    use_common_ground=False,
    run = 1,
    lastPartType = None,
    use_tools = True,
    use_oracle=False,
    num_oracle = None
    
):
    requestCount = 0
    start = time.time()
    target_structure    = structure_data['structure']
    target_spans        = structure_data['spans']          # already int keys
    # target_director_views = structure_data['director_views'] #possible inconsistency: use current live target views from game state:  target_director_views = get_director_views_fn(target_structure, spans=target_spans)
    target_director_views = get_director_views_fn(target_structure, spans=target_spans)
    # ── Init game state ───────────────────────────────────────
    partComplete = True
    game_state = EnhancedGameState(
        target_structure=target_structure,
        target_spans=target_spans,
        partComplete=partComplete,
        partType = lastPartType
    )
  
    # ── Sanity check: stored views vs recomputed ──────────────
    tmp = copy.deepcopy(game_state.current_structure)
    game_state.current_structure = copy.deepcopy(target_structure)
    recomputed = game_state.get_director_views()
    game_state.current_structure = tmp

    # compare against RAW views (before swap) not swapped_dir_views
    raw_views = {
        d: {row: items[::-1] for row, items in rows.items()}
        if d in ["D2", "D3"] else rows
        for d, rows in recomputed.items()
    }
    compare_views("TARGET VIEW CONSISTENCY BEFORE GAME STARTS", target_director_views, raw_views)

    
    conversation_history = []

    game_results = {
        'structure_id'        : structure_data['id'],
        'structure_idx'       : structure_idx,
        'complexity'          : structure_data['complexity'],
        'target_structure'    : target_structure,
        'target_spans'        : {str(k): v for k, v in target_spans.items()},
        'target_director_views': target_director_views,
        'completed'           : False,
        'turns_taken'         : 0,
        'final_progress'      : 0.0,
        'final_structure'     : {},
        'stopping_reason'     : 'max_turns_reached',
        'partialCompletion'   : partComplete,
    }

    if(partComplete):
        lastPartType = game_state.partType
        game_results['partialCompletionCategory'] = game_state.partType

    for d in director_agents.values():
        game_results[d.director_id + " Archetype"] = d.archetype

    game_results['turns'] = []

    # ── MAIN LOOP ─────────────────────────────────────────────
    for turn in range(max_turns):
        print(f"\n--- Turn {turn+1}/{max_turns} ---")
        game_state.increment_turn()
        cValues = {}
        cValues['Turn']=(turn + 1)
        turn_data = {
            'turn_number'     : turn + 1,
            'timestamp'       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'structure_before': copy.deepcopy(game_state.current_structure),
            'spans_before'    : copy.deepcopy(game_state.current_spans),
        }

        try:
            # ── Directors ─────────────────────────────────────
            # Full board state for directors to reason with
            full_board_state = game_state.get_director_views()

            mism, total = view_distance(full_board_state, target_director_views)
            print(f"  View mismatches vs target: {mism}/{total}")

            public_conversation = "\n".join(conversation_history)
            director_order = random.choices(["D1", "D2", "D3"], k=3)
            random.shuffle(director_order)

            director_responses = {}
            for did in director_order:
                print(f"  {did} responding...")
 
                director_prompt = director_agents[did].get_director_prompt(
                    current_view=full_board_state,           # full board, all layers
                    target_view=target_director_views[did],  # partial — director's perspective only
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks
                )
                resp = director_agents[did].generate_response(
                    current_view=full_board_state,           # full board, all layers
                    target_view=target_director_views[did],  # partial — director's perspective only
                    conversation_history=public_conversation,
                    available_blocks=game_state.available_blocks
                )
                requestCount += 1
                
                if(resp["internal_thinking"].__contains__("429")):
                    end = time.time()
                    length = end - start

                    print("\n\n####################################")
                    print(resp["internal_thinking"])
                    print("Time: " + str(length))
                    print("Requests: " + str(requestCount))
                    print("####################################\n\n")
                    sys.exit(1)

                director_responses[did] = resp
                conversation_history.append(f"{did}: {resp['public_message']}")
        
                print(f"  {did}: {resp['public_message'][:80]}")
                turn_data[f'director_prompt_{did}'] = director_prompt
                # print(f"  {did} INTENT: {intent}")
            turn_data['director_responses'] = director_responses
             
            def is_validResponse(val):
                return val[1]['public_message'] != "No message provided"

            # ── Builder ───────────────────────────────────────
            director_discussion_full = "\n".join(
                f"{did}: {r['public_message']}"
                for did, r in filter(is_validResponse, director_responses.items())
            )

          
            oracle_moves = None
            if use_oracle:
                rng = random.Random(structure_idx * 1000 +  (turn + 1))
           
                oracle_moves = get_oracle_moves(game_state, n=num_oracle, rng=rng)
                print(f"  [ORACLE] {len(oracle_moves)} correct moves available this turn with num oracle:{num_oracle}:")
                for m in oracle_moves:
                    span_str = f" → {m['span_to']}" if m.get('span_to') else ""
                    print(f"    {m['action'].upper()} {m.get('block','')} "
                        f"@ {m['position']} layer {m['layer']}{span_str}")
                        
            builder_prompt = builder_agent.get_builder_prompt(
                director_discussion=director_discussion_full,
                current_state=game_state.current_structure,
                available_blocks=game_state.available_blocks,
                use_tools=use_tools,
                oracle_moves=oracle_moves,   
            )
            
            if use_tools:
                builder_move = builder_agent.generate_move_with_tools(
                director_discussion=director_discussion_full,
                game_state=game_state,  # pass state so tool can simulate
                oracle_moves=oracle_moves, 
                check_prompt_tokens=True  # ← only when needed
            )
            else:
                builder_move = builder_agent.generate_move(
                director_discussion=director_discussion_full,
                current_state=game_state.current_structure,  # pass state so tool can simulate
                available_blocks=game_state.available_blocks,
                oracle_moves=oracle_moves, 
            )
            
            print(f"  Builder move: {builder_move}")
            cValues['Transcription'] = director_discussion_full
            cValues['Failed Move']="N/A"
            cValues['Structure Placement']="N/A"
            cValues['Side Placement']="N/A"
            cValues['Overall Structure State']="N/A"
            cValues['Group Agreement']='N/A'
            #requestCount += 1 

            # ── Execute ───────────────────────────────────────
            if builder_move['action'] == 'clarify':
                conversation_history.append(f"Builder: {builder_move['clarification']}")
                turn_data['move_executed'] = False
                turn_data['clarification'] = builder_move['clarification']
                cValues['Clarify'] = True

            else:
                cValues['Clarify'] = "N/A"
                if 'confirmation' in builder_move:
                    conversation_history.append(f"Builder: {builder_move['confirmation']}")
                    turn_data['builder_confirmation'] = builder_move.get('confirmation', '')
                    turn_data['builder_move_raw'] = copy.deepcopy(builder_move)

                success, progress_data, structurePlacement, sidePlacement, overall = game_state.execute_move(builder_move)

                if use_oracle and oracle_moves:
                    attempted = builder_move
                    match = any(
                        attempted.get('action') == m['action'] and
                        attempted.get('position') == m['position'] and
                        attempted.get('layer') == m['layer']
                        for m in oracle_moves
                    )
                    print(f"  [ORACLE CHECK] builder chose from oracle list: {match}")

                    turn_data['oracle_moves'] = [
                        {k: v for k, v in m.items()} for m in oracle_moves
                    ] if oracle_moves else []

                    turn_data['builder_followed_oracle'] = any(
                        builder_move.get('action') == m['action'] and
                        builder_move.get('position') == m['position'] and
                        builder_move.get('layer') == m['layer']
                        for m in oracle_moves
                    ) if oracle_moves else None


                turn_data['builder_prompt'] = builder_prompt
                turn_data['move_attempted'] = copy.deepcopy(builder_move)
                turn_data['move_executed']  = success
                turn_data['progress_data']  = progress_data

                cValues['Failed Move']=(not success)
                turn_data['failed_move']=(not success)
                if(success):
                    turn_data['correct_structure_placement']=structurePlacement
                    turn_data['correct_side_placement']=sidePlacement
                    turn_data['overall_structure_correctness']=overall
                    cValues['Structure Placement']=(structurePlacement)
                    cValues['Side Placement']=(sidePlacement)
                    cValues['Overall Structure State']=(overall)
                cValues['Group Agreement']='N/A'
                turn_data['director_discussion_full'] = director_discussion_full
                # turn_data['director_discussion_used'] = director_discussion_used

                if success:
                    print(f"  Move OK. Progress: {progress_data['metrics']['overall_progress']:.3f}")
                    conversation_history.append(
                        f"Builder: Placed {builder_move['block']} at {builder_move['position']}"
                        + (f"↔{builder_move['span_to']}" if builder_move.get('span_to') else "")
                        + f" layer {builder_move['layer']}. "
                        f"Current board: {json.dumps(game_state.current_structure)}"
                    )
                    if use_common_ground and common_ground_agent is not None:
                        print("Analyzing common ground...")
                        common_ground_result = common_ground_agent.generate_common_ground(
                            director_responses=director_responses,
                            current_board_state=game_state.current_structure,
                            conversation_history="\n".join(conversation_history),
                            last_move=progress_data
                        )
                        turn_data['common_ground'] = common_ground_result
                        agreement = common_ground_result.get('agreement', "No")
                        cValues['Group Agreement'] = agreement
                    else:
                        cValues['Group Agreement'] = "Disabled"
                if not success:
                    err = progress_data.get('error', 'Unknown')
                    msg = f"Move failed: {err}. Please give different instructions."
                    #removing board state as well after builder failed move. too much info in context. 
                    conversation_history.append(
                    f"Builder: FAILED — {err}. Board unchanged: {json.dumps(game_state.current_structure)}"
                )
                    print(f"  Move FAILED: {err}")

            # ── Completion check ──────────────────────────────
            if game_state.is_complete():
                print("  Target achieved!")
                turn_data['target_achieved'] = True
                game_results['stopping_reason'] = 'target_achieved'
                game_results['turns'].append(turn_data)
                break
            else:
                turn_data['target_achieved'] = False
            turn_data['conversation_snapshot'] = list(conversation_history)
            # Trim conversation history
            # conversation_history = conversation_history[-8:] #harsher trimming
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]
             
            game_results['turns'].append(turn_data)

            # New row data as a single-row DataFrame
            new_row_df = pd.DataFrame([cValues])

            # Concatenate the original and new DataFrames
            correctness = pd.concat([correctness, new_row_df], ignore_index=True)
            print(correctness)

        except Exception as e:
            print(f"  Error turn {turn+1}: {e}")
            turn_data['error'] = str(e)
            game_results['turns'].append(turn_data)
            break

    # ── Finalize ──────────────────────────────────────────────
    game_results['turns_taken']     = len(game_results['turns'])
    game_results['final_structure'] = copy.deepcopy(game_state.current_structure)
    game_results['final_spans']     = copy.deepcopy(game_state.current_spans)

    if game_state.progress_tracker.progress_history:
        last = game_state.progress_tracker.progress_history[-1]
        game_results['final_progress'] = last.get('metrics', {}).get('overall_progress', 0.0)
        game_results['completed']      = game_results['final_progress'] >= 0.95

    game_results['progress_summary'] = game_state.get_progress_summary()

    print(f"\nGame done: {game_results['turns_taken']} turns, "
          f"progress={game_results['final_progress']:.3f}, "
          f"completed={game_results['completed']}")

    return game_results, correctness, lastPartType

if __name__ == "__main__":
    import argparse
    load_dotenv()
    experiment = "experiment2"

    parser = argparse.ArgumentParser(description="CRAFT Runner")
    parser.add_argument("--mode",           type=str, default="local",
                        choices=["api", "local"],
                        help="Director mode: 'api' for frontier models, 'local' for open-weight")
    parser.add_argument("--director",       type=str, default=None,
                        help="Specific director model to run (api: model name, local: key from LOCAL_MODELS)")
    parser.add_argument("--builder",        type=str, default="gpt-4o-mini",
                        help="Builder model name")
    parser.add_argument("--dataset",        type=str, default="/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json",
                        help="Path to structures dataset JSON")
    parser.add_argument("--output",         type=str, default=None,
                        help="Output directory (default: auto-generated from builder model)")
    parser.add_argument("--turns",          type=int, default=20,
                        help="Max turns per game")
    parser.add_argument("--run",            type=int, default=3,
                        help="Run index for deterministic seeding")
    parser.add_argument("--oracle",         action="store_true",
                        help="Enable oracle candidate moves for builder")
    parser.add_argument("--oracle_n",       type=int, default=5,
                        help="Number of oracle moves to show per turn")
    parser.add_argument("--no_tools",       action="store_true",
                        help="Disable builder tool use (simulate_move)")
    parser.add_argument("--structures",     type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated structure indices to run (e.g. '0,1,5'). Default: all")
    parser.add_argument("--quantize",       type=str, default=None,
                        choices=["4bit", "8bit"],
                        help="Quantization for local models (qwen-72b/32b default 4bit)")
    parser.add_argument("--max_tokens",     type=int, default=3000,
                        help="Override max output tokens for director (overrides per-model defaults)")
    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')

    # ── Config ────────────────────────────────────────────────
    DIRECTOR_MODE = args.mode
    BUILDER_MODEL = args.builder
    DATASET_PATH  = args.dataset
    MAX_TURNS     = args.turns
    RUN           = args.run
    USE_ORACLE    = args.oracle
    ORACLE_N      = args.oracle_n
    USE_TOOLS     = not args.no_tools
    # ── Auto-generate output dir from config ──────────────────
    oracle_tag = f"oracle{ORACLE_N}" if USE_ORACLE else "no_oracle"
    tools_tag  = "tools" if USE_TOOLS else "no_tools"
    run_tag    = f"run{RUN}"
    print(args)

    OUTPUT_DIR = args.output or (
        f"craft_results/"
        f"{DIRECTOR_MODE}/"
        #f"{oracle_tag}_{tools_tag}_{run_tag}"
        f"{experiment}_{run_tag}"
    )

    LOCAL_MODELS = {
        "mistral-7b":       "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen-7b":          "Qwen/Qwen2.5-7B-Instruct",
        #"llama-8b":         "meta-llama/Llama-3.1-8B-Instruct",
        #"qwen-14b":         "Qwen/Qwen2.5-14B-Instruct",
        #"qwen-32b":         "Qwen/Qwen2.5-32B-Instruct",
        #"gemma-9b":         "google/gemma-2-9b-it",
        #"deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        #"qwen-72b":         "Qwen/Qwen2.5-72B-Instruct",
    }

    API_DIRECTOR_MODELS = [
        "gemini-3-flash-preview",
        "gpt-4o",
        #"gpt-5",
        #"gpt-4o-mini",
        #"gpt-4.1-mini",
        #"gemini-2.5-flash",
        #"gemini-2.5-flash-lite",
        #"gemini-3.1-flash-lite-preview",
        #"claude-haiku-4-5",
        #"claude-sonnet-4-6",
    ]

    MAX_TOKENS_BY_MODEL = {
        "gpt-5":                          2000,
        "gpt-4o-mini":                    2000,
        "gpt-4.1-mini":                   2000,
        "gpt-4o":                         2000,
        "claude-haiku-4-5":               3000,
        "claude-sonnet-4-6":              3000,
        "gemini-2.5-flash":               3000,
        "gemini-2.5-flash-lite":          2000,
        "gemini-3-flash-preview":         2000,
        "gemini-3.1-flash-lite-preview":  2000,
    }
    DEFAULT_MAX_TOKENS = 500

    # ── Structure selection ───────────────────────────────────
    with open(DATASET_PATH, "r") as f:
        n_structures = len(json.load(f))
    print(f"Total structures: {n_structures}")

    if args.structures:
        structure_indices = [int(i) for i in args.structures.split(",")]
    else:
        structure_indices = list(range(n_structures))
    print(f"Running structures: {structure_indices}")

    # ── Model selection ───────────────────────────────────────
    if args.director:
        if DIRECTOR_MODE == "api":
            models_to_run_api = [args.director]
            models_to_run_local = {}
        else:
            if args.director not in LOCAL_MODELS:
                raise ValueError(f"Unknown local model key: {args.director}. Choose from {list(LOCAL_MODELS.keys())}")
            models_to_run_local = {args.director: LOCAL_MODELS[args.director]}
            models_to_run_api = []
    else:
        models_to_run_api = API_DIRECTOR_MODELS
        models_to_run_local = LOCAL_MODELS

    # def get_part_type_for_structure(structure_index, run):
    #     seed = hash((structure_index, run)) % (2**32)
    #     rng  = random.Random(seed)
    #     return rng.choice(EnhancedGameState.PARTIAL_OPTIONS)

    # print("\nPartType sequence for this run:")
    # for i in structure_indices:
    #     pt = get_part_type_for_structure(i, RUN)
    #     print(f"  structure_{i:02d} → {pt}")

    def run_all_structures(director_model_name, shared_model=None, shared_tokenizer=None):
        for structure_index in structure_indices:
            partType = "empty"
            print(f"\n  [{director_model_name}] structure={structure_index}/{n_structures-1} "
                  f"partType={partType}")
            try:
                results, _ = run_craft_experiments(
                    dataset_path=DATASET_PATH,
                    structure_index=structure_index,
                    director_model_name=director_model_name,
                    builder_model_name=BUILDER_MODEL,
                    api_key=api_key,
                    max_turns=MAX_TURNS,
                    output_dir=OUTPUT_DIR,
                    use_common_ground=False,
                    run=RUN,
                    lastPartType=partType,
                    shared_model=shared_model,
                    shared_tokenizer=shared_tokenizer,
                    builder_tool_use=USE_TOOLS,
                    use_oracle=USE_ORACLE,
                    num_oracle=ORACLE_N,
                    max_tokens=args.max_tokens if args.max_tokens else MAX_TOKENS_BY_MODEL.get(director_model_name, DEFAULT_MAX_TOKENS),
                )
                print(f"  Done [{director_model_name}] structure={structure_index} "
                      f"games={len(results['games'])}")
            except Exception as e:
                import traceback
                print(f"  FAILED [{director_model_name}] structure={structure_index}: {e}")
                traceback.print_exc()
                continue

    # ── Run ───────────────────────────────────────────────────
    if DIRECTOR_MODE == "api":
        for model_name in models_to_run_api:
            print(f"\n{'='*60}\nAPI DIRECTOR: {model_name}\n{'='*60}")
            run_all_structures(director_model_name=model_name)

    elif DIRECTOR_MODE == "local":
        for model_name, model_path in models_to_run_local.items():
            print(f"\n{'='*60}\nLOADING MODEL: {model_name}\n{'='*60}")
            try:
                shared_model, shared_tokenizer = load_local_director_pipeline(
                    model_path,
                    quantize=args.quantize or ("4bit" if model_name in ("qwen-72b", "qwen-32b") else None),
                )
                print(f"Loaded: {shared_model}")
            except Exception as e:
                print(f"FAILED to load {model_name}: {e}")
                continue

            run_all_structures(
                director_model_name=model_name,
                shared_model=shared_model,
                shared_tokenizer=shared_tokenizer,
            )

            print(f"\nReleasing {model_name} from memory...")
            try:
                del shared_model, shared_tokenizer
            except Exception:
                pass
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU memory released.\n")

    print("All done.")