#!/bin/bash
python run_craft.py --mode api --builder gpt-5.4-mini --run 1 --director gpt-5.4-mini --builderPrompt Base --no_tools
python run_craft.py --mode api --builder gpt-5.4-mini --run 2 --director gpt-5.4-mini --builderPrompt Base --no_tools

python run_craft.py --mode api --builder gpt-5.4-mini --run 1 --director gpt-5.4-mini --builderPrompt Pragmatic1
python run_craft.py --mode api --builder gpt-5.4-mini --run 2 --director gpt-5.4-mini --builderPrompt Pragmatic1

python run_craft.py --mode api --builder gpt-5.4-mini --run 1 --director gpt-5.4-mini --builderPrompt Pragmatic1 --no_tools
python run_craft.py --mode api --builder gpt-5.4-mini --run 2 --director gpt-5.4-mini --builderPrompt Pragmatic1 --no_tools