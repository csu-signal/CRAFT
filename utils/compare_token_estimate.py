"""Compare TokenEstimator's local estimates against real Claude or OpenAI usage on the same workload.

Each example runs end to end once through the local model and once through the API, so
multi-turn histories grow with each model's own replies. Output error is split into a
tokenizer factor (local tokenizer vs API counts on the API's own text) and a length
factor (local reply lengths vs API reply lengths, both in local tokens).

    python utils/compare_token_estimate.py --out token_compare.json
    python utils/compare_token_estimate.py --input my_examples.jsonl
"""
import argparse
import json
import os
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

from token_estimator import CORRECTION, TokenEstimator


def user(text):
    return {"role": "user", "content": text}


def system(text):
    return {"role": "system", "content": text}


EXAMPLES = [
    {"id": "fact_qa", "messages": [user("What is the capital of Australia, and why isn't it Sydney?")]},
    {"id": "classify", "messages": [
        system("Classify the sentiment of the review as positive, negative, or mixed. Answer with one word."),
        user("The battery lasts forever, but the screen scratches if you look at it wrong.")]},
    {"id": "json_extract", "messages": [
        system("Extract the fields name, date, amount, and currency. Return only JSON."),
        user("Hi, this is Dana Ortiz. I paid the invoice on March 3rd 2025, a total of 1,240.50 euros, "
             "via bank transfer. Please confirm receipt.")]},
    {"id": "summarize", "messages": [user(
        "Summarize the following in three sentences:\n\n"
        "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 "
        "individual reefs and 900 islands stretching over 2,300 kilometres. It is located in the "
        "Coral Sea, off the coast of Queensland, Australia, and can be seen from outer space. The reef "
        "is built by billions of tiny organisms known as coral polyps and supports a wide diversity of "
        "life. It was selected as a World Heritage Site in 1981. Climate change, pollution, crown-of-thorns "
        "starfish outbreaks and fishing are the primary threats to its health, with mass bleaching events "
        "in 2016, 2017, 2020, 2022 and 2024 damaging large portions of shallow-water coral. Tourism to the "
        "reef generates billions of dollars annually for the Australian economy.")]},
    {"id": "code", "messages": [user(
        "Write a Python function that merges overlapping intervals in a list, with a brief explanation.")]},
    {"id": "explain", "messages": [user("Explain how TCP congestion control works to a new-grad engineer.")]},
    {"id": "brainstorm", "messages": [user("Give me 10 name ideas for a quiet neighborhood coffee shop.")]},
    {"id": "translate", "messages": [user(
        "Translate into Spanish and German: 'The meeting has been moved to Thursday afternoon because "
        "half the team is traveling on Wednesday.'")]},
    {"id": "multi_turn_plan", "messages": [
        user("I want to learn linear algebra in 4 weeks for machine learning. Make me a study plan.")],
     "followups": ["Make week 2 more detailed.", "Now compress the whole plan into 5 bullet points."]},
    {"id": "multi_turn_debug", "messages": [
        system("You are a concise senior Python reviewer."),
        user("Why does this return the wrong result?\n\n"
             "def average(xs):\n    total = 0\n    for x in xs:\n        total =+ x\n    return total / len(xs)")],
     "followups": ["Fix it and also handle the empty list.", "Write two pytest tests for the fixed version."]},
]

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get tomorrow's forecast for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

SIMULATE_TOOL = {"type": "function", "function": {
    "name": "simulate_move",
    "description": "Dry-run a proposed move against the environment. Returns {ok, error, metrics, correctness, "
                   "resulting_structure}. Does NOT mutate game state.",
    "parameters": {"type": "object", "properties": {"move": {
        "type": "object",
        "description": "Proposed move with keys: action (place|remove|clarify), block (e.g. 'gs'), "
                       "position (e.g. '(0,0)'), layer (int), span_to (e.g. '(1,0)' or null).",
        "properties": {"action": {"type": "string", "enum": ["place", "remove", "clarify"]},
                       "block": {"type": "string"}, "position": {"type": "string"},
                       "layer": {"type": "integer"}, "span_to": {"type": ["string", "null"]}},
        "required": ["action", "position", "layer"]}}, "required": ["move"]}}}

# Each call to a tool returns its fixed result from "tool_results".
TOOL_EXAMPLES = [
    {"id": "tool_weather", "tools": [WEATHER_TOOL],
     "messages": [user("Should I pack an umbrella for Paris and Tokyo tomorrow? Check the forecast for both.")],
     "tool_results": {"get_weather": json.dumps({"forecast": "light rain", "high_c": 14, "chance_of_rain": 0.7})}},
    {"id": "tool_craft_builder", "tools": [SIMULATE_TOOL],
     "messages": [
         system("You are a Builder agent. You may call simulate_move up to 2 times to dry-run moves. "
                "After simulations, output ONE final move in the PLACE/REMOVE/CLARIFY text format. "
                "No JSON, no extra commentary."),
         user("CURRENT BOARD: empty 3x3 grid, positions (0,0) to (2,2), layers 0-2.\n"
              "AVAILABLE BLOCKS: gs (green small), ys (yellow small), bl (blue large, spans two cells).\n"
              "DIRECTOR DISCUSSION:\n"
              "D1: Put a green small block in the bottom-left corner.\n"
              "D2: Agreed, bottom-left is (0,0) on layer 0.\n"
              "D3: After that, a yellow small block next to it at (1,0).\n\n"
              "OUTPUT FORMAT: PLACE:block_code:position:layer:CONFIRM:reason")],
     "tool_results": {"simulate_move": json.dumps({"ok": True, "blocks_correct": 1, "blocks_total": 9,
                                                   "overall_progress": 0.11})}},
]
MAX_TOOL_ROUNDS = 4

# Sonnet 5.5 rejects "disabled"; "between_tools" still thinks between tool calls in tool examples.
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}


def assistant_message(text, tool_calls):
    return {"role": "assistant", "content": text, **({"tool_calls": tool_calls} if tool_calls else {})}


def call_record(messages, tools, text, tool_calls, input_tokens, output_tokens, truncated, stop_reason):
    # Tool calls are generated output, so they're included in the text used for local counting.
    calls_text = "".join(c["function"]["name"] + c["function"]["arguments"] for c in tool_calls)
    return {"messages": list(messages), "tools": tools, "text": text + calls_text,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "truncated": truncated, "stop_reason": stop_reason}


def claude_messages(messages):
    """OpenAI-style history (including tool calls and results) in Claude's message format."""
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                        "input": json.loads(c["function"]["arguments"])} for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class ClaudeChat:
    """Claude chat callable with the same signature as TokenEstimator.chat that records usage."""

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.calls = []

    def __call__(self, messages, max_tokens=None, tools=None):
        """Reply text, or an OpenAI-style assistant message when `tools` is given."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        kwargs = {"system": system} if system else {}
        if tools:
            kwargs["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                                "input_schema": t["function"]["parameters"]} for t in tools]
        # Thinking off so output tokens are only the reply text, like the local model's.
        response = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            thinking=THINKING_OFF.get(self.model, {"type": "disabled"}),
            messages=claude_messages(messages), **kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = [{"id": b.id, "type": "function",
                       "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                      for b in response.content if b.type == "tool_use"]
        usage = response.usage
        # Billed input is split across uncached, cache-write and cache-read counts.
        input_tokens = (usage.input_tokens + (usage.cache_creation_input_tokens or 0)
                        + (usage.cache_read_input_tokens or 0))
        self.calls.append(call_record(messages, tools, text, tool_calls, input_tokens, usage.output_tokens,
                                      response.stop_reason == "max_tokens", response.stop_reason))
        return assistant_message(text, tool_calls) if tools else text


class OpenAIChat:
    """OpenAI chat callable with the same signature as TokenEstimator.chat that records usage."""

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.calls = []

    def __call__(self, messages, max_tokens=None, tools=None):
        """Reply text, or an OpenAI-style assistant message when `tools` is given."""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_completion_tokens=max_tokens,
            **({"tools": tools} if tools else {}))
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = [{"id": c.id, "type": "function",
                       "function": {"name": c.function.name, "arguments": c.function.arguments}}
                      for c in choice.message.tool_calls or []]
        self.calls.append(call_record(messages, tools, text, tool_calls, response.usage.prompt_tokens,
                                      response.usage.completion_tokens, choice.finish_reason == "length",
                                      choice.finish_reason))
        return assistant_message(text, tool_calls) if tools else text


def run_tool_example(llm, example, max_tokens):
    messages = list(example["messages"])
    for _ in range(MAX_TOOL_ROUNDS):
        reply = llm(messages, max_tokens=max_tokens, tools=example["tools"])
        messages = messages + [reply]
        if not reply.get("tool_calls"):
            return
        messages += [{"role": "tool", "tool_call_id": c["id"],
                      "content": example["tool_results"].get(c["function"]["name"], '{"error": "unknown tool"}')}
                     for c in reply["tool_calls"]]


def run_example(llm, example, max_tokens):
    messages = list(example["messages"])
    reply = llm(messages, max_tokens=max_tokens)
    for followup in example.get("followups", []):
        messages = messages + [{"role": "assistant", "content": reply}, user(followup)]
        reply = llm(messages, max_tokens=max_tokens)


def ratio(a, b):
    return round(a / b, 3) if b else None


def summarize(est_calls, api_calls, rows):
    def total(key, calls):
        return sum(c[key] for c in calls)

    est_in, est_out = total("input_tokens", est_calls), total("output_tokens", est_calls)
    api_in, api_out = total("input_tokens", api_calls), total("output_tokens", api_calls)
    api_out_local = total("local_count_output", api_calls)
    example_ratios = sorted(ratio(r["est_input"] + r["est_output"], r["api_input"] + r["api_output"])
                            for r in rows)
    return {
        "examples": len(rows), "calls": {"est": len(est_calls), "api": len(api_calls)},
        "input_tokens": {"est": est_in, "api": api_in, "ratio": ratio(est_in, api_in),
                         "corrected_ratio": ratio(est_in * CORRECTION, api_in)},
        "output_tokens": {"est": est_out, "api": api_out, "ratio": ratio(est_out, api_out),
                          "corrected_ratio": ratio(est_out * CORRECTION, api_out)},
        "input_tokenizer_ratio": ratio(total("local_count_input", api_calls), api_in),
        "output_tokenizer_ratio": ratio(api_out_local, api_out),
        "output_length_ratio": ratio(est_out, api_out_local),
        "truncated_calls": {"est": total("truncated", est_calls), "api": total("truncated", api_calls)},
        "per_example_total_ratio": {"min": example_ratios[0],
                                    "median": example_ratios[len(example_ratios) // 2],
                                    "max": example_ratios[-1]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path,
                        help='JSONL, one {"id", "messages", optional "followups"} or {"id", "messages", "tools", '
                             '"tool_results"} per line; defaults to built-in examples')
    parser.add_argument("--api-model", default="claude-haiku-5-5",
                        help="claude-* models use the Anthropic API, anything else the OpenAI API")
    parser.add_argument("--api-key-env",
                        help="defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY depending on the model")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("token_compare.json"))
    args = parser.parse_args()

    if args.input:
        lines = args.input.read_text(encoding="utf-8").splitlines()
        examples = [json.loads(line) for line in lines if line.strip()]
        for i, ex in enumerate(examples):
            ex.setdefault("id", str(i))
    else:
        examples = EXAMPLES + TOOL_EXAMPLES

    load_dotenv()
    est = TokenEstimator(max_tokens=args.max_tokens)
    if args.api_model.startswith("claude"):
        key = os.getenv(args.api_key_env or "ANTHROPIC_API_KEY")
        api = ClaudeChat(anthropic.Anthropic(api_key=key), args.api_model)
    else:
        key = os.getenv(args.api_key_env or "OPENAI_API_KEY")
        api = OpenAIChat(openai.OpenAI(api_key=key), args.api_model)

    def est_llm(messages, max_tokens=None, tools=None):
        if tools:
            return est.chat_with_tools(messages, tools, max_tokens=max_tokens)
        return est.chat(messages, max_tokens=max_tokens)

    rows, calls = [], {True: ([], []), False: ([], [])}
    for ex in examples:
        uses_tools = bool(ex.get("tools"))
        run = run_tool_example if uses_tools else run_example
        e0, a0 = len(est.calls), len(api.calls)
        run(est_llm, ex, args.max_tokens)
        run(api, ex, args.max_tokens)
        local_calls, api_calls = est.calls[e0:], api.calls[a0:]
        calls[uses_tools][0].extend(local_calls)
        calls[uses_tools][1].extend(api_calls)
        row = {"id": ex["id"], "tools": uses_tools,
               "calls": {"est": len(local_calls), "api": len(api_calls)}}
        for side, side_calls in (("est", local_calls), ("api", api_calls)):
            row[f"{side}_input"] = sum(c["input_tokens"] for c in side_calls)
            row[f"{side}_output"] = sum(c["output_tokens"] for c in side_calls)
            row[f"{side}_truncated"] = sum(c["truncated"] for c in side_calls)
        rows.append(row)
        print(f"{ex['id']:<20} in {row['est_input']:>6} / {row['api_input']:<6} "
              f"out {row['est_output']:>6} / {row['api_output']:<6} (est / api)")

    for c in api.calls:
        c["local_count_input"] = est.count(c["messages"], c["tools"])
        c["local_count_output"] = est.count(c["text"])

    summary = {"api_model": args.api_model, "max_tokens": args.max_tokens}
    # Plain and tool examples are summarized separately so plain results stay comparable across runs.
    for key, uses_tools in (("plain", False), ("tool_examples", True)):
        subset = [r for r in rows if r["tools"] == uses_tools]
        if subset:
            summary[key] = summarize(*calls[uses_tools], subset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "examples": rows, "est_calls": est.calls,
                                    "api_calls": api.calls}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
