#!/bin/bash
python run_craft.py --mode api --builder gpt-4.1-mini --run 1 --director gpt-5.4-mini --builderPrompt Literal1
python run_craft.py --mode api --builder gpt-4.1-mini --run 2 --director gpt-5.4-mini --builderPrompt Literal1

python run_craft.py --mode api --builder gpt-4.1-mini --run 1 --director gpt-5.4-mini --builderPrompt Literal1 --no_tools
python run_craft.py --mode api --builder gpt-4.1-mini --run 2 --director gpt-5.4-mini --builderPrompt Literal1 --no_tools

python run_craft.py --mode api --builder gpt-4.1-mini --run 1 --director gpt-5.4-mini --builderPrompt Base
python run_craft.py --mode api --builder gpt-4.1-mini --run 2 --director gpt-5.4-mini --builderPrompt Base
