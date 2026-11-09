#!/bin/bash
# GPT-5.4-mini - single full info builder with oracle (N=3) (DEBUG)
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_oracle3_gpt54mini \
#     --backend openai --model gpt-5.4-mini --reasoning-effort none \
#     --oracle-in-prompt --oracle-n 3 --runs 1 --verbose --limit 5 --turns 5

# # GPT-5.4-mini - single full info builder with oracle (N=3)
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_oracle3_gpt54mini \
#     --backend openai --model gpt-5.4-mini --reasoning-effort none \
#     --oracle-in-prompt --oracle-n 3 --runs 1 --verbose

# # GPT-5.6-luna - single full info builder with oracle (N=3)
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_oracle3_gpt56luna \
#     --backend openai --model gpt-5.6-luna --reasoning-effort none \
#     --oracle-in-prompt --oracle-n 3 --runs 1 --verbose

# # GPT-5.4-mini - single full info builder with tool
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_gpt54mini \
#     --backend openai --model gpt-5.4-mini --reasoning-effort none \
#     --sim-tool --runs 1 --verbose

# # GPT-5.6-luna - single full info builder with tool
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_gpt56luna \
#     --backend openai --model gpt-5.6-luna --reasoning-effort none \
#     --sim-tool --runs 1 --verbose

# GPT-5.4-mini - full info directors and builder with tool
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json  --out-dir runs/directors_all_gpt54mini \
#     --backend openai --model gpt-5.4-mini --reasoning-effort none \
#     --directors --director-views all --director-max-tokens 2000 \
#     --sim-tool --runs 1 --verbose

# # GPT-5.6-luna - full info directors and builder with tool
# python run_full_info_builder.py --structures ../data/structures_dataset_20.json  --out-dir runs/directors_all_gpt56luna \
#     --backend openai --model gpt-5.6-luna --reasoning-effort none \
#     --directors --director-views all --director-max-tokens 2000 \
#     --sim-tool --runs 1 --verbose

# GPT-4.1-mini - full info directors and builder with tool
python run_full_info_builder.py --structures ../data/structures_dataset_20.json  --out-dir runs/directors_all_gpt41mini \
    --backend openai --model gpt-4.1-mini --reasoning-effort none \
    --directors --director-views all --director-max-tokens 2000 \
    --sim-tool --runs 1 --verbose

# GPT-4.1-mini - single full info builder with oracle (N=3)
python run_full_info_builder.py --structures ../data/structures_dataset_20.json --out-dir runs/collapsed_oracle3_gpt41mini \
    --backend openai --model gpt-4.1-mini --reasoning-effort none \
    --oracle-in-prompt --oracle-n 3 --runs 1 --verbose