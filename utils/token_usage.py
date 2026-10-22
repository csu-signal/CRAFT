from __future__ import annotations
import argparse
import json
import math
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and \
    value >= 0 else None


def token_totals(records):
    rows = list(records)
    result = {"calls": len(rows), "failed_calls": 0, "calls_missing_usage": 0,
              "input_tokens": 0, "output_tokens": 0, "truncated_calls": 0}
    for row in rows:
        result["failed_calls"] += row.get("status") == "error"
        result["truncated_calls"] += row.get("finish_reason") in ("length", "max_tokens")
        inp, out = _count(row.get("input_tokens")), _count(row.get("output_tokens"))
        result["calls_missing_usage"] += inp is None or out is None
        result["input_tokens"] += inp or 0
        result["output_tokens"] += out or 0
    result["usage_complete"] = result["calls_missing_usage"] == 0
    return result


class UsageTracker:
    def __init__(self, log_path="token_usage.jsonl"):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.lock = threading.Lock()

    def record(self, *, input_tokens=None, output_tokens=None, example_id=None,
               model="unspecified", backend="manual", status="ok", reasoning_tokens=None,
               finish_reason=None, elapsed_seconds=None, **metadata):
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError("example_id must be a string or None")
        if status not in ("ok", "error"):
            raise ValueError("status must be ok or error")
        for value in (input_tokens, output_tokens, reasoning_tokens):
            if value is not None and _count(value) is None:
                raise ValueError("Token counts must be nonnegative Python integers or None")
        row = dict(metadata)
        row.update(timestamp=datetime.now(timezone.utc).isoformat(), model=str(model),
                   backend=str(backend), example_id=example_id, status=status,
                   input_tokens=input_tokens, output_tokens=output_tokens,
                   reasoning_tokens=reasoning_tokens, finish_reason=finish_reason,
                   elapsed_seconds=elapsed_seconds)
        encoded = json.dumps(row, allow_nan=False)
        with self.lock:
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded + "\n")
            self.records.append(row)
        return row

    def summary(self):
        with self.lock:
            return token_totals(self.records)

    def report(self, *, planned_examples=None, input_price_per_million=None,
               output_price_per_million=None, safety_factor=1.0):
        with self.lock:
            rows = list(self.records)
        totals = token_totals(rows)
        groups = {}
        for row in rows:
            if row.get("example_id") is not None:
                groups.setdefault(row["example_id"], []).append(row)
        result = {"models": sorted({r["model"] for r in rows}), "totals": totals,
                  "examples": {key: token_totals(value) for key, value in groups.items()},
                  "near_context_limit_calls": sum(r.get("context_near_limit", False) for r in rows)}
        if not math.isfinite(safety_factor) or safety_factor < 1:
            raise ValueError("safety_factor must be finite and >= 1")
        if planned_examples is not None:
            if not isinstance(planned_examples, int) or isinstance(planned_examples, bool) or planned_examples <= 0:
                raise ValueError("planned_examples must be a positive integer")
            if not rows or any(r.get("example_id") is None for r in rows):
                raise ValueError("Assign an example_id to every call before extrapolating")
            if not totals["usage_complete"] or totals["failed_calls"]:
                raise ValueError("Incomplete usage or failed calls: fix the benchmark before extrapolating")
            scale = planned_examples / len(groups) * safety_factor
            result["estimate"] = {"sample_examples": len(groups),
                                  "planned_examples": planned_examples, "safety_factor": safety_factor,
                                  "input_tokens": math.ceil(totals["input_tokens"] * scale),
                                  "output_tokens": math.ceil(totals["output_tokens"] * scale)}
            prices = (input_price_per_million, output_price_per_million)
            if any(p is not None for p in prices):
                if any(p is None or not math.isfinite(p) or p < 0 for p in prices):
                    raise ValueError("Supply both finite, nonnegative per-million prices")
                est = result["estimate"]
                est["approximate_cost"] = (est["input_tokens"] * prices[0] +
                                            est["output_tokens"] * prices[1]) / 1_000_000
        elif input_price_per_million is not None or output_price_per_million is not None:
            raise ValueError("Cost projection requires planned_examples")
        return result

    def save_report(self, path="token_report.json", **kwargs):
        result = self.report(**kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result



class TokenEstimator:
    def __init__(self, model="qwen2.5:7b-instruct", backend="ollama", base_url=None,
                 log_path="token_usage.jsonl", temperature=0.1, max_tokens=512,
                 num_ctx=8192, timeout=300, api_key=None, tracker=None):
        if backend not in ("ollama", "openai"):
            raise ValueError("backend must be ollama or openai")
        if max_tokens <= 0 or num_ctx <= max_tokens or timeout <= 0:
            raise ValueError("Require 0 < max_tokens < num_ctx and timeout > 0")
        self.model, self.backend = model, backend
        self.base_url = (base_url or ("http://localhost:11434" if backend == "ollama"
                                     else "http://localhost:8000/v1")).rstrip("/")
        self.tracker = tracker if tracker is not None else UsageTracker(log_path)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.num_ctx, self.timeout, self.api_key = num_ctx, timeout, api_key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def records(self):
        return self.tracker.records

    def summary(self):
        return self.tracker.summary()

    def report(self, **kwargs):
        return self.tracker.report(**kwargs)

    def save_report(self, path="token_report.json", **kwargs):
        return self.tracker.save_report(path, **kwargs)

    def chat(self, messages, *, example_id=None, seed=None, return_response=False):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError("example_id must be a string or None")
        if self.backend == "ollama":
            options = {"temperature": self.temperature, "num_predict": self.max_tokens,
                       "num_ctx": self.num_ctx}
            if seed is not None:
                options["seed"] = seed
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "options": options}
            endpoint = "/api/chat"
        else:
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "temperature": self.temperature, "max_tokens": self.max_tokens}
            if seed is not None:
                payload["seed"] = seed
            endpoint = "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "model": self.model, "backend": self.backend, "example_id": example_id,
                  "status": "error", "input_tokens": None, "output_tokens": None,
                  "reasoning_tokens": None, "finish_reason": None}
        started = time.monotonic()
        try:
            request = urllib.request.Request(self.base_url + endpoint, data=data,
                                             headers=headers, method="POST")
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = json.load(response)
            if raw.get("error"):
                raise RuntimeError("Model server returned an error")
            if self.backend == "ollama":
                message = raw["message"]
                inp, out = raw.get("prompt_eval_count"), raw.get("eval_count")
                finish = raw.get("done_reason")
                usage = {"prompt_eval_count": inp, "eval_count": out}
            else:
                choice = raw["choices"][0]
                message = choice["message"]
                usage = raw.get("usage") or {}
                inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
                finish = choice.get("finish_reason")
                record["reasoning_tokens"] = _count(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            text = message.get("content") or ""
            record.update(status="ok", input_tokens=_count(inp), output_tokens=_count(out),
                          finish_reason=finish, raw_usage=usage)
            record["context_near_limit"] = (self.backend == "ollama" and
                                            (record["input_tokens"] or 0) >= self.num_ctx * 0.9)
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 4)
            self.tracker.record(**record)
        return raw if return_response else text


def record_transformers(tracker, inputs, generated, *, example_id=None,
                        model="unspecified", eos_token_id=None, **metadata):
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise ValueError("Helper supports one prompt and one returned sequence; record batches manually")
    width = ids.shape[1]
    if sequences.shape[1] < width or sequences[0, :width].tolist() != ids[0].tolist():
        raise ValueError("Expected decoder-only sequences containing the original prompt prefix")
    mask = inputs.get("attention_mask")
    inp = int(mask.sum().item()) if mask is not None else width
    continuation = sequences[0, width:].tolist()
    eos = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    end = next((i + 1 for i, token in enumerate(continuation) if token in eos), len(continuation))
    return tracker.record(input_tokens=inp, output_tokens=end, example_id=example_id,
                          model=model, backend="transformers", **metadata)


class TransformersAdapter:
    def __init__(self, model, tokenizer, *, tracker=None, model_name=None, device=None,
                 max_new_tokens=512):
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("This adapter supports decoder-only text models")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model, self.tokenizer = model, tokenizer
        self.tracker = tracker if tracker is not None else UsageTracker()
        self.model_name = model_name or getattr(model.config, "_name_or_path", "unspecified")
        self.device = device
        self.max_new_tokens = max_new_tokens

    def chat(self, messages, *, example_id=None, **generation_kwargs):
        import torch

        options = dict(generation_kwargs)
        options.setdefault("max_new_tokens", self.max_new_tokens)
        config = options.get("generation_config", self.model.generation_config)
        for name in ("num_beams", "num_return_sequences"):
            value = options.get(name, getattr(config, name, 1))
            if value != 1:
                raise ValueError("Adapter supports one greedy/sampled sequence, not beam search")
        if options.get("streamer") is not None:
            raise ValueError("Streaming is not supported by this adapter")
        started = time.monotonic()
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            inputs = inputs.to(self.device if self.device is not None else self.model.device)
            self.model.eval()
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **options)
            config = options.get("generation_config", self.model.generation_config)
            eos = options.get("eos_token_id", config.eos_token_id)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            continuation = sequences[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        except Exception as exc:
            self.tracker.record(example_id=example_id, model=self.model_name,
                                backend="transformers", status="error",
                                error_type=type(exc).__name__,
                                elapsed_seconds=round(time.monotonic() - started, 4))
            raise
        eos_ids = eos if isinstance(eos, (list, tuple, set)) else [eos]
        finished = any(token in eos_ids for token in continuation.tolist())
        record_transformers(self.tracker, inputs, generated, example_id=example_id,
                            model=self.model_name, eos_token_id=eos,
                            finish_reason="stop" if finished else "length",
                            elapsed_seconds=round(time.monotonic() - started, 4))
        return text


def main():
    parser = argparse.ArgumentParser(description="Run independent JSONL chat examples on a local model")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--base-url")
    parser.add_argument("--log", default="token_usage.jsonl")
    parser.add_argument("--report", default="token_report.json")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--planned-examples", type=int)
    parser.add_argument("--safety-factor", type=float, default=1.0)
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("id", i)) for i, row in enumerate(examples)]
    if len(ids) != len(set(ids)):
        parser.error("Input example IDs must be unique")
    client = TokenEstimator(model=args.model, backend=args.backend, base_url=args.base_url,
                            log_path=args.log, max_tokens=args.max_tokens, num_ctx=args.num_ctx)
    for example_id, row in zip(ids, examples):
        client.chat(row["messages"], example_id=example_id)
    result = client.save_report(args.report, planned_examples=args.planned_examples,
                                safety_factor=args.safety_factor)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
