"""
Full-game evaluation of an API builder (OpenAI, Gemini or Anthropic).

Usage (run from echo_experiments/):
    python baseline_sanity_check.py --label api_gpt-4o-mini --builder_model gpt-4o-mini --episodes_per_structure 5
    python baseline_sanity_check.py --label api_gpt-5.4 --builder_model gpt-5.4 --episodes_per_structure 5
    python baseline_sanity_check.py --label api_gemini-3.8-flash --builder_model gemini-3.8-flash --episodes_per_structure 5
    python baseline_sanity_check.py --label api_claude-haiku-4.5 --builder_model claude-haiku-4-5 --episodes_per_structure 5

Keys: OPENAI_API_KEY (also used by the directors), GEMINI_API_KEY, ANTHROPIC_API_KEY or CLAUDE_API_KEY.
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_CRAFT_ROOT = Path(__file__).resolve().parent.parent
if str(_CRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CRAFT_ROOT))

from api_builders import default_max_tokens, is_reasoning_model, make_api_generate_fn, provider_for
from eval_harness import add_protocol_args, builder_system_prompt, run_eval

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_protocol_args(parser)
    g = parser.add_argument_group("API builder")
    g.add_argument("--builder_model", type=str, default="gpt-4o-mini")
    g.add_argument("--temperature", type=float, default=0.1,
                   help="falls back to the provider default if the model refuses it")
    g.add_argument("--builder_max_tokens", type=int, default=None,
                   help="output budget incl. hidden reasoning; default 250 (1024 with cot), 8192 for reasoning models")
    g.add_argument("--reasoning_effort", type=str, default=None,
                   help="passed through for models that support it (e.g. low/medium/high); "
                        "Anthropic: output_config.effort")
    g.add_argument("--thinking", type=str, default=None, choices=["disabled", "adaptive"],
                   help="Anthropic only; default is the model's own default (e.g. adaptive on Sonnet 5)")
    args = parser.parse_args()

    max_tokens = args.builder_max_tokens or default_max_tokens(args.builder_model, args.prompt_style)
    print(f"[api-builder] {args.builder_model} via {provider_for(args.builder_model)} "
          f"(reasoning={is_reasoning_model(args.builder_model)}, max_tokens={max_tokens}, "
          f"temperature={args.temperature}, reasoning_effort={args.reasoning_effort})")
    generate_fn, stats = make_api_generate_fn(
        args.builder_model,
        system_prompt_fn=lambda oracle_shown: builder_system_prompt(oracle_shown, args.prompt_style),
        max_tokens=max_tokens, temperature=args.temperature, reasoning_effort=args.reasoning_effort,
        thinking=args.thinking,
    )
    config = {
        "builder_kind": "api", "builder_model": args.builder_model, "provider": provider_for(args.builder_model),
        "temperature": args.temperature, "max_tokens": max_tokens, "reasoning_effort": args.reasoning_effort,
        "thinking": args.thinking,
        "api_stats": stats,
    }
    run_eval(args, generate_fn, config)
    print(f"[api-builder] {stats}")
    if stats["length_stops"]:
        print(f"[api-builder] WARNING: {stats['length_stops']}/{stats['calls']} responses hit the token limit")


if __name__ == "__main__":
    main()
