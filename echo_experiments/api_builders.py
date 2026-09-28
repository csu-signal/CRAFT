"""
API builder adapters (OpenAI, Gemini, Anthropic) returning the
(input_ids, new_tokens, decoded_text) triple run_builder_episode expects.

Provider quirks handled here rather than per model:
  * Reasoning models (GPT-5 family, o-series, Gemini 3 thinking) spend part
    of the output budget on hidden reasoning -- a 250-token cap returns an
    empty answer. Their default budget is larger, and every response that
    stopped on the length limit is counted in `stats` so it can't go
    unnoticed.
  * OpenAI reasoning models reject `max_tokens` (need
    `max_completion_tokens`) and may only accept the default temperature;
    if a temperature is refused, the call is retried without it and the
    run records that the provider default was used.
  * Transient errors (rate limits, 5xx, timeouts) are retried with
    exponential backoff; a call that still fails raises, stopping the run
    so it can be resumed rather than silently scoring a parse failure.
"""
import os
import re
import time

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
    if is_reasoning_model(model):
        return 8192
    return 1024 if prompt_style == "cot" else 250  # 250 = BuilderAgent.generate_move's production value


def _with_retries(fn, what):
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None)
            # 4xx other than rate limiting won't fix itself
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                raise
            if attempt == RETRIES:
                raise
            wait = min(2 ** attempt * 5, 300)
            print(f"  [{what}] {type(e).__name__}: {str(e)[:200]} -- retrying in {wait}s")
            time.sleep(wait)


def make_api_generate_fn(model, system_prompt_fn, max_tokens, temperature=None, reasoning_effort=None):
    """system_prompt_fn(oracle_shown: bool) -> str."""
    provider = provider_for(model)
    stats = {"calls": 0, "length_stops": 0, "empty_responses": 0, "temperature_refused": False,
             "provider": provider}

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
        # max_completion_tokens is accepted by every current OpenAI chat model;
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
        # token tensors are never used for API builders (no gradients); the
        # length carries the output token count into the results
        return torch.zeros(1, dtype=torch.long), torch.zeros(n_out or 0, dtype=torch.long), text

    return generate_fn, stats
