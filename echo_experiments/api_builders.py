"""API builder generate_fns for OpenAI, Gemini, and Anthropic models."""
import os
import re
import time
from collections import deque

import torch

RETRIES = 6

REASONING_PATTERNS = [r"^gpt-5", r"^o\d", r"^gemini-3", r"^gemini-2\.5-pro"]


def provider_for(model):
    name = model.lower()
    if "claude" in name:
        return "anthropic"
    if "gemini" in name:
        return "gemini"
    return "openai"


def is_reasoning_model(model):
    return any(re.search(p, model.lower()) for p in REASONING_PATTERNS)


def default_max_tokens(model, prompt_style):
    # reasoning models and Claude spend output tokens on reasoning before the answer
    if is_reasoning_model(model):
        return 8192
    if provider_for(model) == "anthropic":
        return 2048
    return 1024 if prompt_style == "cot" else 250


# abort if most of the last TRUNCATION_WINDOW responses hit the token limit
TRUNCATION_WINDOW = 20
TRUNCATION_ABORT_RATE = 0.5


def _with_retries(fn, what):
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                raise
            if attempt == RETRIES:
                raise
            wait = min(2 ** attempt * 5, 300)
            print(f"  [{what}] {type(e).__name__}: {str(e)[:200]} -- retrying in {wait}s")
            time.sleep(wait)


def make_api_generate_fn(model, system_prompt_fn, max_tokens, temperature=None, reasoning_effort=None,
                         thinking=None):
    """system_prompt_fn(oracle_shown: bool) -> str. thinking: None (provider default), "disabled" or
    "adaptive" -- Anthropic only."""
    provider = provider_for(model)
    stats = {"calls": 0, "length_stops": 0, "empty_responses": 0, "temperature_refused": False,
             "provider": provider}
    recent_stops = deque(maxlen=TRUNCATION_WINDOW)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"))
    else:
        from openai import OpenAI
        client = (OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                  if provider == "gemini" else OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    def call(system, user):
        if provider == "anthropic":
            kwargs = dict(model=model, system=system, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": user}])
            if thinking:
                kwargs["thinking"] = {"type": thinking}
            if reasoning_effort:
                kwargs["output_config"] = {"effort": reasoning_effort}
            if temperature is not None and not stats["temperature_refused"]:
                kwargs["temperature"] = temperature
            try:
                r = client.messages.create(**kwargs)
            except Exception as e:
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    print(f"  [builder {model}] temperature={temperature} refused; using the provider default")
                    stats["temperature_refused"] = True
                    kwargs.pop("temperature")
                    r = client.messages.create(**kwargs)
                else:
                    raise
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return text, r.stop_reason == "max_tokens", r.usage.output_tokens

        kwargs = dict(model=model, messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": user}])
        # Gemini's OpenAI-compatible endpoint takes max_tokens
        kwargs["max_completion_tokens" if provider == "openai" else "max_tokens"] = max_tokens
        if temperature is not None and not stats["temperature_refused"]:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            r = client.chat.completions.create(**kwargs)
        except Exception as e:
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                print(f"  [builder {model}] temperature={temperature} refused; using the provider default")
                stats["temperature_refused"] = True
                kwargs.pop("temperature")
                r = client.chat.completions.create(**kwargs)
            else:
                raise
        choice = r.choices[0]
        usage = getattr(r, "usage", None)
        return (choice.message.content or ""), choice.finish_reason == "length", \
            (usage.completion_tokens if usage else 0)

    def generate_fn(prompt_text, oracle_moves):
        text, hit_limit, n_out = _with_retries(
            lambda: call(system_prompt_fn(bool(oracle_moves)), prompt_text), f"builder {model}")
        stats["calls"] += 1
        stats["length_stops"] += int(hit_limit)
        stats["empty_responses"] += int(not text.strip())
        if hit_limit and not text.strip():
            print(f"  [builder {model}] WARNING: hit the {max_tokens}-token limit with no visible answer "
                  f"(reasoning budget?) -- raise --builder_max_tokens")
        recent_stops.append(hit_limit)
        if len(recent_stops) == TRUNCATION_WINDOW and sum(recent_stops) / TRUNCATION_WINDOW > TRUNCATION_ABORT_RATE:
            raise SystemExit(
                f"[builder {model}] {sum(recent_stops)} of the last {TRUNCATION_WINDOW} responses hit the "
                f"{max_tokens}-token limit -- the model is being cut off before it answers. Rerun with a larger "
                f"--builder_max_tokens or --thinking disabled (and --resume to keep the episodes already saved)")
        # placeholder tensors; new_tokens length carries the output token count
        return torch.zeros(1, dtype=torch.long), torch.zeros(n_out or 0, dtype=torch.long), text

    return generate_fn, stats
