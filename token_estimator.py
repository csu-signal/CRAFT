"""Estimate LLM API token usage by running a pipeline against Qwen2.5-14B-Instruct locally.

    est = TokenEstimator(max_tokens=1024)
    for example in sample:
        run_pipeline(example, llm=est.chat)
    est.show()
"""
import json
import threading
from types import SimpleNamespace

MODEL = "Qwen/Qwen2.5-14B-Instruct"
MESSAGE_OVERHEAD = 4
REPLY_OVERHEAD = 3
# claude's tokenizer produces about 1.5x as many tokens as Qwen's for the same text.
CORRECTION = 1.5


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


def _field(obj, name):
    """Attribute or key, so Gemini configs and contents work as SDK objects or plain dicts."""
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _gemini_text(x):
    """Text of a Gemini string, Part, Content, or list of them."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "".join(_gemini_text(item) for item in x)
    parts = _field(x, "parts")
    return _gemini_text(parts) if parts is not None else _field(x, "text") or ""


class TokenEstimator:
    """Chat function backed by a local Qwen model that records token counts per call."""

    def __init__(self, max_tokens=512, device=None, temperature=0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(device).eval()
        self.max_tokens, self.temperature = max_tokens, temperature
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.calls = []
        # Pipelines may call chat() from several threads; generation on one model runs one at a time.
        self.lock = threading.Lock()

    def count(self, x):
        """Token count of a string, or of a list of chat messages including formatting overhead."""
        if isinstance(x, str):
            return len(self.tokenizer.encode(x, add_special_tokens=False))
        return sum(self.count(_text(m.get("content"))) + MESSAGE_OVERHEAD for m in x) + REPLY_OVERHEAD

    def chat(self, messages, max_tokens=None):
        """Generate a reply to OpenAI-style messages and record its token usage."""
        return self._chat(messages, max_tokens)[0]

    def _chat(self, messages, max_tokens):
        """Reply text and this call's usage record."""
        limit = max_tokens or self.max_tokens
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        if self.temperature > 0:
            sampling = {"do_sample": True, "temperature": self.temperature}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=limit,
                                      pad_token_id=self.tokenizer.pad_token_id, **sampling)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.eos_ids
        if stopped:
            new = new[:-1]
        call = {"input_tokens": self.count(messages), "output_tokens": len(new), "truncated": not stopped}
        with self.lock:
            self.calls.append(call)
        return self.tokenizer.decode(new, skip_special_tokens=True), call

    def report(self):
        """Token totals over all recorded calls, scaled by CORRECTION."""
        return {"calls": len(self.calls),
                "truncated_calls": sum(c["truncated"] for c in self.calls),
                "input_tokens": round(sum(c["input_tokens"] for c in self.calls) * CORRECTION),
                "output_tokens": round(sum(c["output_tokens"] for c in self.calls) * CORRECTION)}

    def show(self):
        print(json.dumps(self.report(), indent=2))


class EstimatorClient:
    """Stands in for an OpenAI, Anthropic or Gemini client and answers chat calls with a TokenEstimator.

    Swap it in wherever code keeps its client, e.g. `agent.client = EstimatorClient(est)`.
    Responses report raw Qwen token counts in `usage` (`usage_metadata` for Gemini).
    """

    def __init__(self, est):
        self.est = est
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._openai_create))
        self.messages = SimpleNamespace(create=self._anthropic_create)
        self.models = SimpleNamespace(generate_content=self._gemini_generate)

    def _openai_create(self, messages, max_tokens=None, max_completion_tokens=None, **_):
        text, call = self.est._chat(messages, max_tokens or max_completion_tokens)
        message = SimpleNamespace(role="assistant", content=text, tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=call["input_tokens"], completion_tokens=call["output_tokens"],
                                total_tokens=call["input_tokens"] + call["output_tokens"])
        choice = SimpleNamespace(index=0, message=message,
                                 finish_reason="length" if call["truncated"] else "stop")
        return SimpleNamespace(choices=[choice], usage=usage)

    def _anthropic_create(self, messages, max_tokens=None, system=None, **_):
        history = [{"role": "system", "content": _text(system)}] if system else []
        history += [{"role": m["role"], "content": _text(m["content"])} for m in messages]
        text, call = self.est._chat(history, max_tokens)
        usage = SimpleNamespace(input_tokens=call["input_tokens"], output_tokens=call["output_tokens"],
                                cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return SimpleNamespace(role="assistant", content=[SimpleNamespace(type="text", text=text)],
                               stop_reason="max_tokens" if call["truncated"] else "end_turn", usage=usage)

    def _gemini_generate(self, contents, config=None, **_):
        system = _field(config, "system_instruction")
        history = [{"role": "system", "content": _gemini_text(system)}] if system else []
        for item in contents if isinstance(contents, list) else [contents]:
            role = _field(item, "role")
            # Strings and bare parts without a role belong to the current user turn.
            if role is None and history and history[-1]["role"] == "user":
                history[-1]["content"] += _gemini_text(item)
            else:
                history.append({"role": "assistant" if role == "model" else "user",
                                "content": _gemini_text(item)})
        text, call = self.est._chat(history, _field(config, "max_output_tokens"))
        candidate = SimpleNamespace(
            content=SimpleNamespace(role="model", parts=[SimpleNamespace(text=text, thought=False)]),
            finish_reason="MAX_TOKENS" if call["truncated"] else "STOP")
        usage = SimpleNamespace(prompt_token_count=call["input_tokens"],
                                candidates_token_count=call["output_tokens"], thoughts_token_count=0,
                                total_token_count=call["input_tokens"] + call["output_tokens"])
        return SimpleNamespace(text=text, candidates=[candidate], usage_metadata=usage)
