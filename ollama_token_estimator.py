"""Ollama-backed version of TokenEstimator (token_estimator.py).

Same interface and report, so EstimatorClient and est.show() work unchanged, but
replies come from a model served by Ollama: no 30 GB download, no large GPU.

    from token_estimator import EstimatorClient
    from ollama_token_estimator import OllamaTokenEstimator

    est = OllamaTokenEstimator()           # qwen2.5:14b-instruct by default
    agent.client = EstimatorClient(est)
    run_pipeline(sample)
    est.show()

Setup:
    ollama pull qwen2.5:14b-instruct
    ollama serve

Token counts
  * Input tokens are counted exactly as TokenEstimator counts them (Qwen tokenizer
    plus the same per-message overhead), so the two backends report comparable
    numbers. Only the tokenizer files are downloaded (a few MB; needs
    `pip install transformers`, not torch). Without transformers the count falls
    back to Ollama's prompt_eval_count, which can come out LOW: Ollama leaves
    out prompt tokens it reuses from its cache.
  * Output tokens are Ollama's eval_count; a call that hits the output cap counts
    as truncated, as in TokenEstimator.
  * Prompts are never cut: each request sets Ollama's context size (num_ctx) to
    fit the prompt plus the output cap, starting at `num_ctx` and only growing
    (each growth makes Ollama reload the model once).
"""
import json
import math
import threading
import urllib.error
import urllib.request

from token_estimator import TokenEstimator, MODEL as TOKENIZER_MODEL, _text

DEFAULT_OLLAMA_MODEL = "qwen2.5:14b-instruct"
DEFAULT_HOST = "http://localhost:11434"


class OllamaTokenEstimator(TokenEstimator):
    """TokenEstimator whose replies come from Ollama instead of transformers."""

    def __init__(self, model=DEFAULT_OLLAMA_MODEL, host=DEFAULT_HOST, max_tokens=512,
                 temperature=0.0, seed=0, num_ctx=16384, timeout=600.0, check=True):
        # Deliberately not calling TokenEstimator.__init__: it loads the 14B model.
        self.ollama_model = model
        self.host = host.rstrip("/").removesuffix("/v1")
        self.max_tokens, self.temperature, self.seed = max_tokens, temperature, seed
        self.num_ctx, self.timeout = num_ctx, timeout
        self.calls = []
        self.lock = threading.Lock()
        if check:
            self._preflight()
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            self.input_count_source = "qwen_tokenizer"
        except Exception as e:  # noqa: BLE001
            self.tokenizer = None
            self.input_count_source = "ollama_prompt_eval_count"
            print(f"[estimate] Qwen tokenizer unavailable ({e}); input tokens will use Ollama's "
                  f"prompt_eval_count, which can undercount. `pip install transformers` fixes this.")
        print(f"[estimate] Ollama estimator: {model} at {self.host}, inputs counted with "
              f"{self.input_count_source}, num_ctx starts at {num_ctx}")

    def _preflight(self):
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as resp:
                names = {m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])}
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"[estimate] Cannot reach Ollama at {self.host} ({e}). Start it with: ollama serve")
        if self.ollama_model not in names and f"{self.ollama_model}:latest" not in names:
            raise SystemExit(f"[estimate] Ollama model '{self.ollama_model}' is not pulled. "
                             f"Run: ollama pull {self.ollama_model}")

    def _prompt_len(self, messages):
        """Exact prompt length as Qwen's chat template renders it (None without a tokenizer)."""
        if self.tokenizer is None:
            return None
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return len(ids["input_ids"] if isinstance(ids, dict) else ids)

    def _chat(self, messages, max_tokens):
        """Reply text and this call's usage record, in TokenEstimator's format."""
        limit = max_tokens or self.max_tokens
        msgs = [{"role": m.get("role", "user"), "content": _text(m.get("content"))} for m in messages]

        # Size the context so the prompt is never cut. Without a tokenizer, assume
        # ~3 characters per token, which errs on the large side.
        prompt_len = self._prompt_len(msgs)
        needed = (prompt_len if prompt_len is not None else
                  sum(len(m["content"]) for m in msgs) // 3) + limit + 256
        with self.lock:
            if needed > self.num_ctx:
                self.num_ctx = int(math.ceil(needed / 2048) * 2048)
                print(f"[estimate] prompt needs {needed} tokens; raising num_ctx to {self.num_ctx}")
            num_ctx = self.num_ctx

        options = {"num_predict": limit, "temperature": self.temperature, "num_ctx": num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        body = json.dumps({"model": self.ollama_model, "messages": msgs,
                           "stream": False, "options": options}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())

        text = (out.get("message") or {}).get("content", "") or ""
        call = {
            "input_tokens": self.count(msgs) if self.tokenizer is not None
            else int(out.get("prompt_eval_count") or 0),
            "output_tokens": int(out.get("eval_count") or 0),
            "truncated": out.get("done_reason") == "length",
            "ollama_prompt_eval_count": out.get("prompt_eval_count"),
            "num_ctx": num_ctx,
        }
        with self.lock:
            self.calls.append(call)
        return text, call

    def report(self):
        """TokenEstimator's totals (scaled by CORRECTION), plus which backend produced them."""
        r = super().report()
        r.update(backend="ollama", model=self.ollama_model, input_count_source=self.input_count_source)
        return r
