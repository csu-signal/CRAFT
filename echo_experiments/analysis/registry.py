"""
What gets analyzed and how it's shown. This is the one file to touch when
adding a baseline/method or a metric -- plots and tables read everything
from here.

  * New method/baseline: add a Method to METHODS. Its `key` is matched as a
    prefix of the eval label (`echo_step350` -> "echo", `api_gpt-4o-mini` ->
    "api"); any `_step<N>` suffix is ignored for naming. Position in the list fixes its color
    slot and row order, so append rather than insert to keep existing
    figures' colors stable.
  * New metric: compute it per episode in eval_results.episode_metrics
    (so it lands in every run's JSON), then add a Metric here.
"""
import re
from dataclasses import dataclass, field

from .style import SERIES, NEUTRAL


@dataclass(frozen=True)
class Method:
    key: str
    display: str
    color: str
    marker: str = "o"
    kind: str = "trained"  # "trained" or "baseline"
    name_fmt: str = "{model} + {name}"
    aliases: tuple = ()  # other label prefixes that mean this method


@dataclass(frozen=True)
class Metric:
    key: str                         # column name in the per-episode results
    display: str                     # axis/table header text
    higher_is_better: bool | None    # None = descriptive only (no "best", no arrow)
    percent: bool = False            # show as % (value is a 0-1 rate)
    bounds: tuple | None = None      # fixed axis range in raw units, e.g. (0, 1)
    decimals: int = 3
    binary: bool = False             # one 0/1 outcome per episode -> Wilson CI
    diagnostic: bool = False         # run-health metric: in the tables, not in the default figures


METHODS = [
    Method("echo", "ECHO", "#127362", "s"),
    Method("api", "API", SERIES[3], "o", kind="baseline", name_fmt="{model}"),
    Method("episode_return", "GRPO", "#c65a66", "o", aliases=("grpo",)),
    Method("rloo_per_turn", "RLOO", "#d9a441", "o", aliases=("rloo",)),
    Method("cot", "CoT", "#d88bb0", "o", kind="baseline"),
    Method("base", "zero-shot", "#a9a9a9", "o", kind="baseline", name_fmt="{model}"),
]


MODEL_STYLES = {
    "Claude Sonnet 5": "#34507f",
    "Claude Sonnet 4.6": "#607fb0",
    "GPT-5.4 mini": "#8a7fb5",
    "GPT-4.1 mini": "#7897bc",
    # listed largest first: this order is also the bar/legend order
    "Qwen2.5-72B 4-bit": "#f2c230",  # yellow
    "Qwen2.5-32B 4-bit": "#eb9834",  # orange
    "Qwen2.5-14B": "#a86b32",
    "Qwen2.5-7B": "#a9a9a9",
}
METHODS_BY_KEY = {k: m for m in METHODS for k in (m.key, *m.aliases)}

METRICS = [
    Metric("final_progress", "Final progress", True, percent=True, bounds=(0, 1), decimals=1),
    Metric("completed", "Completion rate", True, percent=True, bounds=(0, 1), decimals=1, binary=True),
    Metric("oracle_match_rate", "Oracle match", True, percent=True, bounds=(0, 1), decimals=1),
    Metric("invalid_move_rate", "Invalid moves", False, percent=True, bounds=(0, 1), decimals=1),
    Metric("parse_failure_rate", "Parse failures", False, percent=True, bounds=(0, 1), decimals=1),
    Metric("clarify_rate", "Clarify rate", None, percent=True, bounds=(0, 1), decimals=1),
    Metric("progress_gain", "Progress gained", True, percent=True, bounds=(0, 1), decimals=1),
    Metric("correct_move_rate", "Correct moves", True, percent=True, bounds=(0, 1), decimals=1),
    Metric("episode_length", "Episode length", False, decimals=1),
    Metric("reward_mean", "Reward / turn", True, diagnostic=True),
    Metric("director_failure_rate", "Director API failures", False, percent=True, decimals=2, diagnostic=True),
    Metric("completion_tokens_mean", "Builder output tokens", None, decimals=0, diagnostic=True),
]
METRICS_BY_KEY = {m.key: m for m in METRICS}

# Shown when no --metrics are given: the headline panel of the paper figure.
PRIMARY_METRICS = ["final_progress", "completed", "oracle_match_rate", "invalid_move_rate"]

_UNKNOWN = Method("?", "?", NEUTRAL, "X", kind="baseline")
_STEP_RE = re.compile(r"^(?P<prefix>.+?)_step(?P<step>\d+)$")


def short_model_name(hf_id, quantize=None):
    """"Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B"; "gpt-5.4" -> "GPT-5.4";
    "claude-haiku-4-5" -> "Claude Haiku 4.5"; 4-bit runs are marked."""
    if not hf_id:
        return None
    name = re.sub(r"-Instruct$", "", hf_id.rstrip("/").split("/")[-1])
    low = name.lower()
    if low.startswith("gpt-"):
        name = "GPT-" + name[4:].replace("-mini", " mini").replace("-nano", " nano")
    elif low.startswith(("gemini-", "claude-")):
        # claude-haiku-4-5 -> Claude Haiku 4.5; gemini-3.8-flash -> Gemini 3.8 Flash
        parts = re.sub(r"(\d)-(\d)", r"\1.\2", name).split("-")
        name = " ".join(p.capitalize() if p.isalpha() else p for p in parts)
    return f"{name} {quantize.replace('bit', '-bit')}" if quantize else name


def model_size_b(name):
    m = re.search(r"(\d+(?:\.\d+)?)B\b", name or "")
    return float(m[1]) if m else float("inf")


@dataclass(frozen=True)
class Condition:
    """One row in every table/figure."""
    label: str
    method: Method = field(compare=False)
    step: int | None = None
    model: str | None = field(default=None, compare=False)

    @property
    def display(self):
        if self.model is None:
            return self.method.display
        return self.method.name_fmt.format(model=self.model, name=self.method.display)

    @property
    def color(self):
        # zero-shot base sizes and API models are coloured per model; trained
        # methods (all Qwen2.5-7B) per method
        if self.method.kind == "baseline" and self.model in MODEL_STYLES:
            return MODEL_STYLES[self.model]
        return self.method.color

    @property
    def is_ours(self):
        return self.method.key == "echo"

    @property
    def sort_key(self):
        order = METHODS.index(self.method) if self.method in METHODS else len(METHODS)
        within = list(MODEL_STYLES).index(self.model) if self.model in MODEL_STYLES else len(MODEL_STYLES)
        return (order, within, -model_size_b(self.model), self.step if self.step is not None else -1, self.label)


def parse_label(label, model=None):
    """`model` is the display model name (see short_model_name)."""
    m = _STEP_RE.match(label)
    prefix, step = (m["prefix"], int(m["step"])) if m else (label, None)
    # longest matching key wins, so e.g. "episode_return" beats a future "episode"
    for key in sorted(METHODS_BY_KEY, key=len, reverse=True):
        if prefix == key or prefix.startswith(key + "_"):
            return Condition(label, METHODS_BY_KEY[key], step, model)
    print(f"[analysis] warning: label {label!r} matches no registered Method -- shown in neutral; "
          f"add it to analysis/registry.py:METHODS")
    return Condition(label, Method(prefix, prefix, _UNKNOWN.color, _UNKNOWN.marker, _UNKNOWN.kind), step, model)
