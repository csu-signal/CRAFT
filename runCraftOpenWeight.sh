#!/bin/bash
python run_craft.py --mode local --builder gpt-5.4-mini --run 1 --director qwen-3.5-9b --builderPrompt Pragmatic1
python run_craft.py --mode local --builder gpt-4.1-mini --run 1 --director qwen-3.5-9b --builderPrompt Pragmatic1
