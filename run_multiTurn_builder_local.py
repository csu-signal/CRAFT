#!/usr/bin/env python3
"""
run_multiTurn_builder_local.py
==============================

Offline / no-API-cost version of `run_single_builder.py`, running BOTH arms of
each divergence turn through the same local Builder, and now rolling each arm
forward for H turns (default 3) instead of one.

The Builder and the generated directors can also run on a Claude model over
the Anthropic API (--backend anthropic); see "Backends" below.

Rollout design
--------------
For every divergence turn t the harness runs three arms:

  * counterfactual arm — step 0 uses the utterance of the director the POLICY
                         chose
  * factual arm        — step 0 uses the utterance of the director HISTORY
                         chose (`turn["builderSelected"]`)
  * random arm         — RANDOM CONTROL: the Builder hears one director per
                         turn, drawn uniformly at random (see "Random control
                         arm" below). Runs by default; --no-random-arm skips it.

Each arm then continues for steps 1..H-1 (turns t+1 .. t+H-1) on ITS OWN board
and its own conversation. Two independent switches control those turns:

  --followon-source generate (default)
      New D1/D2/D3 utterances every follow-on turn, from the DirectorAgent
      prompt and parser (imported from --director-module, served by the
      director backend), conditioned on the arm's board, each director's
      target view, and the arm's own conversation. Archetypes come from
      CRAFT's D*_archetype so the directors keep their historical
      personalities. Directors speak in order D1, D2, D3 and each sees who
      spoke before it. This gives a continuous causal chain from the
      divergence onward.
  --followon-source history
      Replay the logged CRAFT D1-D3 messages for turn t+k.

  --followon-select all (default)
      The Builder hears every non-silent director, as in the original game.
  --followon-select best
      The Builder hears each director alone; the move with the highest
      whole-board IoU vs target is kept. This uses the target, so it is an
      ORACLE selection rule. history+best isolates compounding returns from
      director selection, with no generation in the loop.

  --followon-select policy  (needs --policy-path)
      The trained index policy picks ONE director per follow-on turn and the
      Builder hears only that director, as at the divergence turn. Scoring
      reuses the policy project's own code (features.encode_arm_state and
      models.ArmStateEvaluator, imported from --policy-code-dir): each
      director's ArmState history up to the current turn is encoded, the three
      rows are scored, and the argmax is taken, exactly as in training. Each
      arm keeps its own histories, so the policy reacts to that arm's board
      and conversation. By default the FACTUAL arm then uses 'all'; set
      --factual-followon-select policy to run the policy in both arms, which
      isolates the step-0 decision under policy control afterwards.

      Before trusting a policy rollout, run --policy-check-only: it rebuilds
      the policy input at every divergence turn and checks that the argmax
      reproduces the logged 'predicted' director. Near 100% means the features
      match training; well below means FEATURE_MODE/FEATURE_TIER,
      USE_POOLED_HISTORY or turn numbering differ from how the .pt was trained.

  --followon-select recommend  (needs --policy-path)
      Policy as RECOMMENDATION rather than filter. The policy scores D1-D3
      exactly as in 'policy' mode, but the Builder hears ALL speaking
      directors, and the policy's pick is appended to the Builder prompt
      (--recommend-template, default "Prioritize D2's instruction. Use the
      other directors' messages to resolve ambiguity."). With
      --recommend-step0 recommend (default), the divergence turn works the
      same way for BOTH arms: each hears all three historical messages plus a
      recommendation naming its own director (policy's pick vs history's),
      so the arms' step-0 inputs differ ONLY in the recommendation.
      --recommend-step0 single keeps the original single-director step 0.
      The factual arm defaults to 'all' after step 0, as with 'policy'.

Random control arm
------------------
The random arm separates the effect of WHICH director the policy picks from
the effect of simply giving the Builder less to sort through. It mirrors the
counterfactual arm's format exactly, but every director choice is a uniform
random draw from the directors who spoke instead of the policy's pick:

  * counterfactual 'policy' (and any non-recommend rule): the random arm hears
    ONE randomly drawn director per follow-on turn.
  * counterfactual 'recommend': the random arm hears all speaking directors
    plus a recommendation naming a randomly drawn director (step 0 too, when
    --recommend-step0 recommend), so it controls for the recommendation's
    content rather than its presence.

--random-step0 sets the random arm's divergence-turn director:
  nonhistory (default) uniform over the speaking directors OTHER than
             history's. This mirrors the counterfactual arm, whose policy pick
             differs from history on every divergence turn by construction;
             the two arms then differ only in who made the choice.
  random     uniform over all speaking directors (can equal history's).
  policy     the policy's logged pick, so step 0 is identical to the
             counterfactual arm and only the follow-on selection rule differs.

Draws are seeded from --random-seed and the decision's identity (game, turn,
arm, step), so they are reproducible, independent of turn order, and identical
on --resume. Paired deltas are reported for counterfactual − factual (as
before), random − factual, and counterfactual − random; the last one is the
policy's value over random selection with the same information restriction.

In every mode the ONLY difference between the arms is the step-0 director
choice; everything downstream is the consequence of that choice.

Backends
--------
  --backend ollama (default)
      Builder and directors served by a local Ollama model through its
      OpenAI-compatible endpoint (--base-url). Seeded and free.
  --backend anthropic
      Builder and directors served by a Claude model through the Anthropic
      Messages API. Needs `pip install anthropic` and ANTHROPIC_API_KEY (or
      --anthropic-api-key). Default model: claude-sonnet-5-5.
  --director-backend {ollama,anthropic}
      Serve the generated directors from a different backend than the Builder
      (e.g. Claude Builder, local directors). Defaults to --backend.

  Things that differ on the Claude API:
    * No sampling seed. Two arms with identical prompts can still differ, so
      compare `builder_prompt_sha1` before attributing a difference to the
      step-0 choice.
    * Newer Claude models reject non-default temperature. The script sends
      --temperature / --director-temperature, and if the API rejects them it
      drops them for the rest of the run (logged, and listed under
      api_usage.params_dropped in the summary). The same happens if the
      installed SDK refuses the keyword (TypeError, before any request is
      sent). --no-send-temperature skips the attempt.
    * Adaptive thinking counts against max_tokens, so the defaults rise to
      4096 for Claude (400/500 for Ollama). You pay only for tokens generated.
      --effort {low,medium,high,xhigh,max} sets the effort level where the
      model supports it (dropped automatically if not).
    * Rate limits / overload (429/529) are retried by the SDK
      (--api-max-retries); --min-request-interval adds a fixed gap between
      calls if you keep hitting limits.
    * Token usage per role (builder / director) is written to the summary.

Ground-truth checks
-------------------
At every step each arm's board is compared against:

  1. the target structure (`satisfaction_full`, as before), and
  2. the historical board after the same turn in CRAFT (the ground-truth
     trajectory): `iou_vs_ground_truth_board`, plus
     `progress_vs_ground_truth = satisfaction_full - ground_truth_satisfaction_full`,
     i.e. how far ahead of / behind what actually happened the arm is.

Caveats that matter for interpretation
--------------------------------------
  * Follow-on utterances were produced in response to the HISTORICAL board, not
    the arm's board. The further an arm drifts from history, the less well the
    instructions fit it. That is inherent to replaying logged dialogue.
  * CRAFT `oracle_moves` for turn t+k were enumerated for the historical board,
    so with --use-oracle alone they are only passed at step 0 (where both arms
    share the historical pre-move board). With --oracle-every-turn, step 0
    still uses the logged CRAFT candidates, and every follow-on turn
    regenerates candidates for that arm's own board with the CRAFT project's
    agents/oracle.py (reconstruct_state + enumerate_correct_actions, flag 'ok' only).
    Step 0 is also regenerated once and compared with the logged candidates
    ('oracle_check'), which validates the regeneration.
  * Rollouts stop early when the game ends (no CRAFT row for the next turn).
    Later steps are therefore computed on a shrinking subset of turns; the
    per-step `n` in the summary shows this. Compare step 2 against step 0
    restricted to the same turns (`final_full_horizon`) rather than pooled.
  * Cost: 3 arms x H steps LLM calls per divergence turn (2 with
    --no-random-arm), plus 3 director calls per arm per follow-on step with
    --followon-source generate. The script prints an upper bound on the number
    of calls before starting. Smoke-test with --limit first.

Everything else (`create_builder_prompt`, `parse_builder_response`,
`_place_block`, `_remove_block`, `getParticalView`, `calculate_iou_board`) is
imported from `run_single_builder` rather than copied, so the two harnesses
cannot drift apart.

Setup
-----
    # Local:
    ollama pull qwen2.5:7b-instruct
    OLLAMA_CONTEXT_LENGTH=8192 ollama serve

    # Claude API:
    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage
-----
    # 1) Check which CRAFT field holds the follow-on director discussion:
    python run_multiTurn_builder_local.py --divergence-dir ... --structures ... \
        --out-dir ... --inspect-row

    # 2) Run (output folder is named automatically; --out-dir overrides):
    python run_multiTurn_builder_local.py \
        --divergence-dir .../divergenceData_policy_seed77_b0.9_kl0.1_closs0.01_nosamp_hpt0.4/validation \
        --structures    .../CRAFT/data/structures_dataset_20.json \
        --model         qwen2.5:7b-instruct \
        --oracle-every-turn --horizon 5 --limit 50 \
        --followon-source generate --followon-select policy \
        --policy-path   .../exports/policy_seed77_b0.9_kl0.1_closs0.01_nosamp_hpt0.4.pt

    # -> .../phase1_refactor/multiTurn/local_qwen2.5-7b-instruct__policy_seed77_b0.9_kl0.1_
    #    closs0.01_nosamp_hpt0.4__validation__unfiltered__oracleEvery__h5_n50__generate-policy

    # 3) Same run with Claude as Builder and directors:
    python run_multiTurn_builder_local.py ... --backend anthropic --model claude-sonnet-5-5
    # -> .../multiTurn/claude_claude-sonnet-5-5__policy_seed77_...

    # The random control arm runs automatically; to skip it:
    python run_multiTurn_builder_local.py ... --no-random-arm

    # See exactly what the Builder receives (first 2 turns printed, all steps saved):
    python run_multiTurn_builder_local.py ... --limit 3 --print-prompts 2 --save-prompts all

    # Interrupted? Rerun the SAME command with --resume: finished turns are skipped
    # (results are saved after every turn) and the summary covers all of them.
    python run_multiTurn_builder_local.py ... --resume

    # Claude Builder, local Ollama directors:
    python run_multiTurn_builder_local.py ... --backend anthropic \
        --director-backend ollama --director-model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from datasets import load_dataset

# Reuse the real prompt/parse/physics/scoring code -- do not reimplement it.
from run_single_builder import (
    BuilderAgent,
    BuilderType,
    _place_block,
    _remove_block,
    calculate_iou_board,
    getParticalView,
    normalize_structure,
)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
BACKENDS = ("ollama", "anthropic")
DEFAULT_MODEL_BY_BACKEND = {"ollama": DEFAULT_MODEL, "anthropic": DEFAULT_CLAUDE_MODEL}
# Builder / director max_tokens when not given on the command line. Claude's
# adaptive thinking draws on the same budget, so it needs more headroom.
DEFAULT_MAX_TOKENS = {"ollama": (400, 500), "anthropic": (4096, 4096)}
AVAILABLE_BLOCKS = ["gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"]
DIRECTORS = ("D1", "D2", "D3")

SILENT_MARKERS = {"", "no message provided", "none", "null", "n/a"}
MOVE_PREFIXES = ("PLACE:", "REMOVE:", "CLARIFY:")

# Candidate CRAFT column names. The first one present in the dataset is used.
# Run with --inspect-row to see the actual columns; override the utterance
# field with --followon-utterance-field if auto-detection picks the wrong one.
UTTERANCE_KEYS = (
    "director_discussion", "selected_utterance", "builder_selected_utterance",
    "director_utterance", "utterance", "discussion", "dialogue",
)
# Special value for --followon-utterance-field: build the follow-on discussion
# from the per-director columns D1_message / D2_message / D3_message, formatted
# "D1: ...\nD2: ...\nD3: ..." with silent directors left out. This is the
# default whenever those three columns exist (as in craft-benchmark-lean).
DIRECTOR_MESSAGES = "directors"
BOARD_AFTER_KEYS = ("structure_after", "structureAfter", "board_after", "gt_structure_after")
BOARD_BEFORE_KEYS = ("structure_before", "structureBefore", "board_before")


# --------------------------------------------------------------------------- #
# Chat backends (shared by the Builder and the generated directors)
# --------------------------------------------------------------------------- #
def _new_usage() -> Dict[str, int]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "max_tokens_stops": 0, "refusals": 0, "errors": 0}


class ChatBackend:
    """
    Minimal chat transport. `complete` takes a system prompt and a list of
    user/assistant messages (alternating, starting with user) and returns the
    model's text. Usage is tracked per role ('builder' / 'director').
    """
    name = "base"
    supports_seed = False

    def __init__(self):
        self.usage: Dict[str, Dict[str, int]] = defaultdict(_new_usage)

    def complete(self, model: str, system: str, messages: List[Dict], temperature: float,
                 max_tokens: int, seed: Optional[int] = None, role: str = "builder") -> str:
        raise NotImplementedError

    def usage_summary(self) -> Dict:
        return {role: dict(u) for role, u in self.usage.items()}


class OllamaBackend(ChatBackend):
    """Ollama's OpenAI-compatible endpoint (the original transport)."""
    name = "ollama"
    supports_seed = True

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "ollama",
                 timeout: float = 300.0):
        super().__init__()
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, model, system, messages, temperature, max_tokens, seed=None, role="builder"):
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}] + list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        u = self.usage[role]
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception:
            u["errors"] += 1
            raise
        u["calls"] += 1
        usage = getattr(completion, "usage", None)
        if usage is not None:
            u["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            u["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        if not completion or not completion.choices:
            return ""
        choice = completion.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            u["max_tokens_stops"] += 1
        return (choice.message.content or "").strip()


class AnthropicBackend(ChatBackend):
    """
    Claude over the Anthropic Messages API.

    * The client is created on first use, so --policy-check-only and
      --preview-director-prompt work without an API key.
    * temperature / effort are sent if requested; if the model rejects one
      with a 400, it is dropped for the rest of the run and logged.
    * Only text blocks are returned (thinking blocks are skipped).
    """
    name = "anthropic"
    supports_seed = False

    def __init__(self, api_key: Optional[str] = None, timeout: float = 300.0,
                 max_retries: int = 8, send_temperature: bool = True,
                 effort: Optional[str] = None, min_interval: float = 0.0):
        super().__init__()
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None
        self._sdk = None
        self.send_temperature = send_temperature
        self.effort = effort
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0
        self.params_dropped: Dict[str, str] = {}
        self._warned_max_tokens = False

    @property
    def sdk(self):
        if self._sdk is None:
            try:
                import anthropic  # noqa: WPS433
            except ImportError:
                sys.exit("[anthropic] The 'anthropic' package is not installed. Run:  pip install anthropic")
            self._sdk = anthropic
        return self._sdk

    def api_key(self) -> Optional[str]:
        return self._api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def client(self):
        if self._client is None:
            key = self.api_key()
            if not key:
                sys.exit("[anthropic] No API key. Set ANTHROPIC_API_KEY or pass --anthropic-api-key.")
            self._client = self.sdk.Anthropic(api_key=key, timeout=self._timeout,
                                              max_retries=self._max_retries)
        return self._client

    def _throttle(self) -> None:
        if self.min_interval:
            wait = self._last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.time()

    def _drop(self, param: str, exc: Exception) -> None:
        self.params_dropped[param] = str(exc)[:300]
        print(f"  [anthropic] model rejected '{param}'; dropping it for the rest of the run. "
              f"({str(exc)[:160]})")

    def complete(self, model, system, messages, temperature, max_tokens, seed=None, role="builder"):
        kwargs = dict(model=model, system=system, messages=list(messages), max_tokens=max_tokens)
        if self.send_temperature and temperature is not None and "temperature" not in self.params_dropped:
            kwargs["temperature"] = temperature
        if self.effort and "effort" not in self.params_dropped:
            # Sent through extra_body so this works across SDK versions.
            kwargs["extra_body"] = {"output_config": {"effort": self.effort}}

        u = self.usage[role]
        while True:
            self._throttle()
            try:
                resp = self.client.messages.create(**kwargs)
                break
            except self.sdk.BadRequestError as exc:
                msg = str(exc).lower()
                if "temperature" in kwargs and "temperature" in msg:
                    self._drop("temperature", exc)
                    kwargs.pop("temperature")
                    continue
                if "extra_body" in kwargs and ("effort" in msg or "output_config" in msg):
                    self._drop("effort", exc)
                    kwargs.pop("extra_body")
                    continue
                u["errors"] += 1
                raise
            except TypeError as exc:
                # Some SDK versions don't accept these keywords at all; the
                # call never reaches the API, so dropping them costs nothing.
                msg = str(exc)
                if "temperature" in kwargs and "'temperature'" in msg:
                    self._drop("temperature", exc)
                    kwargs.pop("temperature")
                    continue
                if "extra_body" in kwargs and "'extra_body'" in msg:
                    self._drop("effort", exc)
                    kwargs.pop("extra_body")
                    continue
                u["errors"] += 1
                raise
            except Exception:
                u["errors"] += 1
                raise

        u["calls"] += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            u["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            u["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        stop = getattr(resp, "stop_reason", None)
        if stop == "max_tokens":
            u["max_tokens_stops"] += 1
            if not self._warned_max_tokens:
                self._warned_max_tokens = True
                print(f"  [anthropic] a {role} response hit max_tokens={max_tokens}. If outputs come "
                      f"back empty or cut off, raise --max-tokens / --director-max-tokens or lower --effort.")
        elif stop == "refusal":
            u["refusals"] += 1
        return "".join(
            getattr(b, "text", "") for b in (resp.content or []) if getattr(b, "type", None) == "text"
        ).strip()

    def usage_summary(self) -> Dict:
        out = super().usage_summary()
        out["params_dropped"] = dict(self.params_dropped)
        return out


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_ollama(base_url: str, model: str, timeout: float = 10.0) -> None:
    """Fail loudly and early rather than after 400 turns of CLARIFY."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"[preflight] Cannot reach Ollama at {root} ({exc}).\n"
            f"            Start it with:  OLLAMA_CONTEXT_LENGTH=8192 ollama serve"
        )

    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        sys.exit(
            f"[preflight] Model '{model}' is not pulled. Available: {sorted(names) or '(none)'}\n"
            f"            Run:  ollama pull {model}"
        )
    print(f"[preflight] Ollama reachable at {root}; model '{model}' present.")


def preflight_anthropic(backend: AnthropicBackend, model: str) -> None:
    """Check the key and the model id before spending anything."""
    if not backend.api_key():
        sys.exit("[preflight] No Anthropic API key. Set ANTHROPIC_API_KEY or pass --anthropic-api-key.")
    sdk = backend.sdk
    try:
        backend.client.models.retrieve(model)
    except sdk.AuthenticationError as exc:
        sys.exit(f"[preflight] Anthropic API key rejected ({exc}).")
    except sdk.NotFoundError:
        sys.exit(f"[preflight] Claude model '{model}' not found. Check the model id "
                 f"(e.g. claude-sonnet-5-5, claude-haiku-5-5, claude-opus-5-5).")
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] Could not verify model '{model}' ({type(exc).__name__}: {exc}); continuing.")
        return
    print(f"[preflight] Anthropic API reachable; model '{model}' available.")


def preflight(backend: ChatBackend, model: str, base_url: str) -> None:
    if isinstance(backend, AnthropicBackend):
        preflight_anthropic(backend, model)
    else:
        preflight_ollama(base_url, model)


def warn_if_prompt_may_be_truncated(prompt: str, assumed_ctx: int = 4096) -> None:
    est = len(prompt) // 4
    if est > assumed_ctx * 0.75:
        print(
            f"[warn] builder prompt ~{est} tokens. If OLLAMA_CONTEXT_LENGTH is at its "
            f"default (~{assumed_ctx}) the prompt may be TRUNCATED. "
            f"Restart the server with OLLAMA_CONTEXT_LENGTH=8192."
        )


# --------------------------------------------------------------------------- #
# Local builder agent
# --------------------------------------------------------------------------- #
def extract_move_line(text: str) -> Optional[str]:
    """Pull the PLACE/REMOVE/CLARIFY line out of a local model's response."""
    if not text:
        return None

    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().strip("[]").strip()
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line).strip()
        line = re.sub(r"^(?:[-*>]\s+|\d+[.)]\s+)", "", line)
        line = line.strip("*").strip()
        upper = line.upper()
        for prefix in MOVE_PREFIXES:
            if upper.startswith(prefix):
                return prefix + line[len(prefix):]

    match = re.search(r"\b(PLACE|REMOVE|CLARIFY):", cleaned, flags=re.IGNORECASE)
    if match:
        tail = cleaned[match.start():].splitlines()[0].strip().strip("[]").strip().rstrip("*")
        return match.group(1).upper() + tail[len(match.group(1)):]
    return None


# FIX: these clarifications are produced by this script, not by the model.
# "Error in move generation" used to be counted as a genuine CLARIFY, which
# hid every exception inside generate_move from the parse-failure stats.
NON_MODEL_CLARIFY_PREFIXES = (
    "Could not parse response", "Parse error",
    "Error in move generation", "Empty response from local model", "No response from local model",
)


def is_parse_failure(move: Dict) -> bool:
    """A real CLARIFY is a valid answer; a parser giving up (or crashing) is not."""
    if move.get("action") != "clarify":
        return False
    text = str(move.get("clarification", ""))
    return text.startswith(NON_MODEL_CLARIFY_PREFIXES)


# --------------------------------------------------------------------------- #
# Oracle adherence
# --------------------------------------------------------------------------- #
def _norm_pos_str(s) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        try:
            parts = [int(p.strip()) for p in s.strip("()").split(",")]
            return "(" + ",".join(str(p) for p in parts) + ")"
        except ValueError:
            return s
    return s


def _coerce_layer(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def move_matches_candidate(move: Dict, candidate_move: Dict) -> bool:
    if move.get("action") != candidate_move.get("action"):
        return False
    if _norm_pos_str(move.get("position")) != _norm_pos_str(candidate_move.get("position")):
        return False
    if _coerce_layer(move.get("layer")) != _coerce_layer(candidate_move.get("layer")):
        return False
    if _norm_pos_str(move.get("span_to")) != _norm_pos_str(candidate_move.get("span_to")):
        return False
    if move.get("action") == "place" and move.get("block") != candidate_move.get("block"):
        return False
    return True


def _candidate_move_dict(cand: Dict) -> Dict:
    if "action" in cand:
        return cand
    return cand.get("move", cand)


def builder_candidates(cands: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """FIX: flat {action, block, position, layer, span_to} dicts, the shape the
    CRAFT oracle_moves (and BuilderAgent.create_builder_prompt) use. Regenerated
    candidates come wrapped as {"move": {...}, "flag": ..., "overall_progress": ...};
    passing those to create_builder_prompt raised KeyError('action') before the
    model was ever called, on every follow-on step."""
    if cands is None:
        return None
    out = []
    for c in cands:
        m = _candidate_move_dict(c)
        if isinstance(m, dict) and "action" in m:
            out.append({k: m.get(k) for k in ("action", "block", "position", "layer", "span_to")})
    return out


def compute_oracle_adherence(move: Dict, oracle_moves: Optional[List[Dict]]) -> Optional[bool]:
    if not oracle_moves:
        return None
    if move.get("action") == "clarify":
        return None
    return any(move_matches_candidate(move, _candidate_move_dict(cand)) for cand in oracle_moves)


class LocalBuilderAgent(BuilderAgent):
    """BuilderAgent served by a ChatBackend (Ollama or the Claude API)."""

    def __init__(
        self,
        backend: ChatBackend,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 400,
        seed: Optional[int] = 0,
        max_retries: int = 2,
    ):
        # Deliberately not calling BuilderAgent.__init__ (it binds to api.openai.com).
        self.backend = backend
        self.client = None  # all generation goes through self.backend
        self.model_name = model_name
        self.oracle_moves = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.last_raw_text: str = ""
        self.last_attempts: int = 0
        self.last_prompt: str = ""
        self.last_system: str = ""
        self.last_error: Optional[str] = None

    def _chat(self, system: str, messages: List[Dict]) -> str:
        return self.backend.complete(
            self.model_name, system, messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
            seed=self.seed, role="builder",
        )

    def generate_move(
        self,
        director_discussion,
        current_state,
        available_blocks,
        oracle_moves=None,
        check_prompt_tokens: bool = False,
        builder_note: Optional[str] = None,
    ) -> Dict:
        self.last_raw_text = ""
        self.last_attempts = 0
        self.last_prompt = ""
        self.last_system = ""
        self.last_error = None
        oracle_moves = builder_candidates(oracle_moves)  # FIX: defensive, see builder_candidates
        try:
            prompt = self.create_builder_prompt(
                director_discussion,
                current_state,
                available_blocks,
                oracle_moves=oracle_moves,
                builderPromptAddin=BuilderType.Base,
            )
            if builder_note:
                # 'recommend' mode: the policy's pick goes at the END of the
                # Builder prompt, after the directors' messages and candidates.
                prompt = f"{prompt}\n\n{builder_note}"
            self.last_prompt = prompt
            if check_prompt_tokens and self.backend.name == "ollama":
                warn_if_prompt_may_be_truncated(prompt)

            system_content_oracle = (
                "You are a Builder in a collaborative LEGO task. "
                "You have been given VERIFIED CANDIDATE MOVES — you MUST choose exactly one from the list. "
                "Respond in the specified PLACE/REMOVE/CLARIFY format. "
                "In your CONFIRM field, write 2-3 sentences: which director(s) you followed, "
                "whether others agreed or conflicted, and why you chose this candidate."
            )
            system_content_base = (
                "You are a Builder in a collaborative LEGO task. "
                "Respond in the specified PLACE/REMOVE/CLARIFY format. "
                "In your CONFIRM field, write 2-3 sentences: which director(s) you are following, "
                "what the other directors said and whether they agreed or conflicted, "
                "and why you chose this move."
            )
            system_content = system_content_oracle if oracle_moves else system_content_base
            system_content += (
                " Output EXACTLY ONE line and nothing else: no preamble, no markdown, "
                "no code fences, no explanation after the line."
            )

            self.last_system = system_content
            messages = [{"role": "user", "content": prompt}]

            last_move: Dict = {"action": "clarify", "clarification": "No response from local model"}
            for attempt in range(self.max_retries + 1):
                self.last_attempts = attempt + 1
                raw = self._chat(system_content, messages)
                self.last_raw_text = raw
                if not raw:
                    last_move = {"action": "clarify", "clarification": "Empty response from local model"}
                    continue

                line = extract_move_line(raw)
                if line is not None:
                    move = self.parse_builder_response(line)
                    if not is_parse_failure(move):
                        return move
                    last_move = move
                else:
                    last_move = {
                        "action": "clarify",
                        "clarification": f"Could not parse response: {raw[:200]}",
                    }

                if attempt < self.max_retries:
                    print(f"  [retry {attempt + 1}] unparseable output, re-prompting for format")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That was not in the required format. Reply with ONE line only, "
                            "starting with PLACE:, REMOVE: or CLARIFY:, using exactly:\n"
                            "PLACE:block_code:position:layer:CONFIRM:interpretation\n"
                            "PLACE:block_code:position:layer:span_to:CONFIRM:interpretation\n"
                            "REMOVE:position:layer:CONFIRM:interpretation\n"
                            "REMOVE:position:layer:span_to:CONFIRM:interpretation\n"
                            "CLARIFY:your question\n"
                            "No other text."
                        ),
                    })

            return last_move

        except Exception as exc:  # noqa: BLE001
            # FIX: keep the traceback; a bare KeyError message ("'action'") is not diagnosable.
            self.last_error = traceback.format_exc()
            print(f"  [builder error] {type(exc).__name__}: {exc}\n{self.last_error}")
            return {"action": "clarify", "clarification": f"Error in move generation: {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_divergence_turns(folder: Path) -> List[Dict]:
    turns: List[Dict] = []
    files = sorted(folder.glob("*.json"))
    if not files:
        sys.exit(f"[data] No divergence logs found in {folder}")
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            turns.extend(json.load(fh))
    print(f"[data] Loaded {len(turns)} divergence turns from {len(files)} files.")
    return turns


def normalize_director_model(name: str) -> str:
    director = name.replace("-", "").lower()
    if director == "deepseeklite":
        director = "deepseekv2lite"
    return director


def director_label_from_combo(model_combo: str) -> str:
    return model_combo.split("+")[0].strip()


def load_decidable_keys(path: Path) -> set:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {
        (rec["turn_number"], rec["structure_id"], normalize_director_model(rec["director_model"]))
        for rec in records if rec["decidable"]
    }


def build_craft_index(dataset_name: str) -> Dict:
    ds = load_dataset(dataset_name)
    split = ds["train"]
    index: Dict = {}
    for row in split:
        key = (row["turn_number"], row["structure_id"], normalize_director_model(row["director_model"]))
        index.setdefault(key, row)
    print(f"[data] Indexed {len(index)} CRAFT rows from '{dataset_name}'.")
    return index


def is_silent(message_obj: Dict) -> bool:
    if not isinstance(message_obj, dict):
        return True
    utterance = (message_obj.get("utterance") or "").strip().lower()
    return utterance in SILENT_MARKERS


def parse_board(value):
    """Boards arrive as dicts/lists, JSON strings, or Python-repr strings."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def first_present(row: Dict, keys) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def inspect_craft_row(index: Dict) -> None:
    """Print one CRAFT row's columns so the follow-on fields can be verified."""
    row = next(iter(index.values()))
    print("\n[inspect] CRAFT row columns (values truncated):")
    for k, v in row.items():
        s = str(v).replace("\n", "\\n")
        print(f"  {k:<32} {s[:140]}{'...' if len(s) > 140 else ''}")
    print(f"\n[inspect] auto-detected follow-on utterance field: {detect_followon_field(row)}")
    print(f"[inspect] auto-detected board-after field:         {first_present(row, BOARD_AFTER_KEYS)}")
    print(f"[inspect] auto-detected board-before field:        {first_present(row, BOARD_BEFORE_KEYS)}")
    if detect_followon_field(row) == DIRECTOR_MESSAGES:
        print("\n[inspect] example follow-on discussion fed to the Builder:")
        print(followon_utterance(row, DIRECTOR_MESSAGES))


def has_director_message_columns(row: Dict) -> bool:
    return all(f"{d}_message" in row for d in DIRECTORS)


def detect_followon_field(row: Dict) -> Optional[str]:
    if has_director_message_columns(row):
        return DIRECTOR_MESSAGES
    return first_present(row, UTTERANCE_KEYS)


def followon_utterance(row: Dict, field: str) -> Optional[str]:
    if field == DIRECTOR_MESSAGES:
        lines = []
        for d in DIRECTORS:
            msg = row.get(f"{d}_message")
            if isinstance(msg, (list, dict)):
                msg = json.dumps(msg)
            msg = (str(msg) if msg is not None else "").strip()
            if msg.lower() in SILENT_MARKERS:
                continue
            lines.append(f"{d}: {msg}")
        return "\n".join(lines) or None

    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        value = json.dumps(value)
    value = str(value).strip()
    return value or None


def ground_truth_board_after(
    index: Dict, turn: int, structure_id, director_model: str, after_key: Optional[str]
):
    """
    Historical board after `turn`. Prefer an explicit after-column on the row;
    otherwise use the NEXT turn's before-board. None if neither exists
    (typically the last turn of a game with no after-column).
    """
    row = index.get((turn, structure_id, director_model))
    if row is not None and after_key and row.get(after_key) not in (None, ""):
        return parse_board(row[after_key])
    nxt = index.get((turn + 1, structure_id, director_model))
    if nxt is not None:
        before_key = first_present(nxt, BOARD_BEFORE_KEYS)
        if before_key:
            return parse_board(nxt[before_key])
    return None


# --------------------------------------------------------------------------- #
# Oracle move regeneration (--oracle-every-turn)
# --------------------------------------------------------------------------- #
TARGET_SPAN_KEYS = ("target_spans", "spans", "structure_spans", "targetSpans")


def target_spans_for(struct_entry: Dict):
    """Target spans from the structures file entry, or None if absent."""
    for k in TARGET_SPAN_KEYS:
        if k in struct_entry and struct_entry[k] is not None:
            v = struct_entry[k]
            return json.loads(v) if isinstance(v, str) else v
    return None


def structure_has_large_blocks(structure: Dict) -> bool:
    return any(str(b).endswith("l") for stack in structure.values() for b in stack)


def oracle_move_key(cand: Dict):
    """Hashable identity of an oracle candidate (same fields as adherence)."""
    m = _candidate_move_dict(cand)
    action = m.get("action")
    return (
        action,
        _norm_pos_str(m.get("position")),
        _coerce_layer(m.get("layer")),
        _norm_pos_str(m.get("span_to")),
        m.get("block") if action == "place" else None,
    )


def usable_candidates(cands: Optional[List[Dict]]) -> List[Dict]:
    """Keep runnable candidates only (flag 'ok', or no flag field at all)."""
    return [c for c in (cands or []) if c.get("flag", "ok") == "ok"]


class OracleRegenerator:
    """
    Recomputes CRAFT oracle candidates for an arbitrary board by calling the
    CRAFT project's own oracle.reconstruct_state + enumerate_correct_actions,
    so follow-on turns get candidates that are valid for THAT arm's board.
    Only flag == 'ok' candidates are passed to the Builder.
    """

    def __init__(self):
        # CRAFT keeps oracle.py at agents/oracle.py; a top-level oracle.py is
        # accepted as a fallback. Either way, run from the CRAFT root so
        # oracle.py's own imports (agents.environment, agents.builder_tools,
        # structure_generator_v2) resolve.
        errors = []
        module = None
        for name in ("agents.oracle", "oracle"):
            try:
                module = importlib.import_module(name)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        if module is None or not hasattr(module, "enumerate_correct_actions"):
            sys.exit("[oracle] Could not import oracle.py (" + "; ".join(errors) + "). Run from the "
                     "CRAFT root, where agents/oracle.py, agents/environment.py, "
                     "agents/builder_tools.py and structure_generator_v2.py are importable.")
        print(f"[oracle] using {module.__name__} ({getattr(module, '__file__', '?')})")
        self._reconstruct = module.reconstruct_state
        self._enumerate = module.enumerate_correct_actions

    def candidates(self, board, spans, target, target_spans) -> Tuple[List[Dict], List[Dict]]:
        # enumerate_correct_actions prints [ENUM DEBUG] lines for every failed
        # simulation; the flags are kept in the record instead.
        with contextlib.redirect_stdout(io.StringIO()):
            state = self._reconstruct(
                {"structure_before": copy.deepcopy(board), "spans_before": copy.deepcopy(spans or {})},
                {"target_structure": copy.deepcopy(target), "target_spans": copy.deepcopy(target_spans or {})},
            )
            all_cands = self._enumerate(state)
        return usable_candidates(all_cands), all_cands


def span_recorded(spans, pos, partner, layer) -> bool:
    """True if a large block's span (pos<->partner at layer) is in `spans`."""
    if not isinstance(spans, dict):
        return False
    entries = spans.get(layer, spans.get(str(layer), []))
    want = {_norm_pos_str(pos), _norm_pos_str(partner)}
    for e in entries or []:
        try:
            if {_norm_pos_str(str(e[0])), _norm_pos_str(str(e[1]))} == want:
                return True
        except (TypeError, IndexError):
            continue
    return False


_SPAN_WARNED = False

# Builder prompt logging, set from --save-prompts / --print-prompts in main().
#   save:  "step0" (default) | "all" | "none" -- which steps keep the full prompt
#   print: True while the current divergence turn is within --print-prompts
PROMPT_LOG = {"save": "step0", "print": False}


# --------------------------------------------------------------------------- #
# Step / rollout execution
# --------------------------------------------------------------------------- #
def _keep_prompt(step_idx: int) -> bool:
    mode = PROMPT_LOG["save"]
    return mode == "all" or (mode == "step0" and step_idx == 0)


def execute_step(
    agent: LocalBuilderAgent,
    arm: str,
    step_idx: int,
    director: Optional[str],
    utterance: str,
    board,
    spans,
    target_norm: Dict,
    oracle_moves,
    check_tokens: bool = False,
    builder_note: Optional[str] = None,
) -> Tuple[Dict, object, object]:
    """
    One Builder move on `board`. Returns (result_dict, board_after, spans_after).
    `board` and `spans` belong to this arm only; the caller deep-copies them at
    the start of the rollout so arms never share state.
    """
    board_before = copy.deepcopy(board)
    oracle_moves = builder_candidates(oracle_moves)  # FIX: flat candidates for prompt, adherence and the record

    move = agent.generate_move(
        director_discussion=utterance,
        current_state=copy.deepcopy(board),
        available_blocks=AVAILABLE_BLOCKS,
        oracle_moves=oracle_moves,
        check_prompt_tokens=check_tokens,
        builder_note=builder_note,
    )
    raw_text, attempts = agent.last_raw_text, agent.last_attempts
    if PROMPT_LOG["print"] and agent.last_prompt:
        bar = "-" * 72
        print(f"\n    {bar}\n    BUILDER PROMPT  [{arm} t+{step_idx}]\n    {bar}")
        print(f"    SYSTEM: {agent.last_system}\n")
        print("\n".join("    " + line for line in agent.last_prompt.splitlines()))
        print(f"    {bar}\n")
    print(f"    [{arm:<14} t+{step_idx}] {director or 'history'}: {move}")

    # NOTE: spans is carried forward as the same object. This assumes
    # _place_block/_remove_block update spans in place (as they do the board).
    # If they instead return updated spans in the third slot, use it here.
    success = False
    if move["action"] == "place":
        success, board, _ = _place_block(
            board, spans, move["position"], move["block"], move["layer"], move["span_to"]
        )
    elif move["action"] == "remove":
        success, board, _ = _remove_block(
            board, spans, move["position"], move["layer"], move["span_to"]
        )

    # Regenerated oracle moves read `spans`; if _place_block does not record a
    # new large block's span there, later removals of it would be mis-enumerated.
    span_ok = None
    if success and move["action"] == "place" and str(move.get("block", "")).endswith("l") \
            and move.get("span_to"):
        global _SPAN_WARNED
        span_ok = span_recorded(spans, move["position"], move["span_to"], move["layer"])
        if not span_ok and not _SPAN_WARNED:
            _SPAN_WARNED = True
            print("  [warn] a placed large block's span was not recorded in `spans` -- "
                  "_place_block may not update spans in place; regenerated oracle moves for "
                  "removing it will be flagged missing_span_info.")

    oracle_adherent = compute_oracle_adherence(move, oracle_moves)
    partial_views = {
        d: calculate_iou_board(normalize_structure(getParticalView(d, board)), target_norm)
        for d in DIRECTORS
    }

    result = {
        "step": step_idx,
        "director": director,
        "utterance": utterance,
        "structure_before": board_before,
        "structure_after": copy.deepcopy(board),
        "satisfaction_full_before": calculate_iou_board(normalize_structure(board_before), target_norm),
        "builder_move": move,
        "move_executed": success,
        "parse_failed": is_parse_failure(move),
        "oracle_moves": copy.deepcopy(oracle_moves) if oracle_moves is not None else None,
        "oracle_candidate_count": len(oracle_moves) if oracle_moves else 0,
        "oracle_adherent": oracle_adherent,
        "span_recorded": span_ok,
        "satisfaction_full": calculate_iou_board(normalize_structure(board), target_norm),
        "satisfaction_partial": partial_views.get(director) if director in DIRECTORS else None,
        "partial_views": partial_views,
        "llm_attempts": attempts,
        "raw_model_text": raw_text,
        # FIX: audit trail. The hash shows whether two arms' Builder inputs
        # differ; the full prompt is kept for the divergence step (step 0).
        "builder_prompt_sha1": hashlib.sha1(
            (agent.last_system + "\n" + agent.last_prompt).encode("utf-8")).hexdigest() if agent.last_prompt else None,
        "builder_prompt": (agent.last_prompt if _keep_prompt(step_idx) else None),
        "builder_system_prompt": (agent.last_system if _keep_prompt(step_idx) else None),
        "builder_error": agent.last_error,
        "builder_note": builder_note,
    }
    return result, board, spans


# --------------------------------------------------------------------------- #
# Index policy (follow-on selection by the trained ArmStateEvaluator)
# --------------------------------------------------------------------------- #
POLICY_MODULES = ("config", "arm_state", "features", "models")


def find_policy_code_dir(policy_path: Path, given: Optional[Path]) -> Path:
    candidates = [given] if given else [policy_path.parent, policy_path.parent.parent, Path.cwd()]
    for c in candidates:
        if c and (Path(c) / "features.py").exists() and (Path(c) / "config.py").exists():
            return Path(c).resolve()
    sys.exit(f"[policy] Could not find features.py/config.py near {policy_path}. "
             f"Pass --policy-code-dir <phase1_refactor folder>.")


class PolicyScorer:
    """
    Loads the trained index policy and scores D1/D2/D3 exactly as training did:
    each director's ArmState history up to the current turn goes through the
    project's own features.encode_arm_state, the three rows go through
    ArmStateEvaluator(X, role_ids), and the pick is the argmax.

    The policy project's config/features/models/arm_state modules are imported
    from --policy-code-dir. If another module named 'config' is already loaded
    (e.g. from the CRAFT code), it is set aside during the import and restored
    afterwards, so neither side sees the other's config.
    """

    def __init__(self, policy_path: Path, code_dir: Path, feature_mode: Optional[str] = None,
                 feature_tier: Optional[str] = None, allow_silent: bool = False):
        if feature_mode:
            os.environ["FEATURE_MODE"] = feature_mode
        if feature_tier:
            os.environ["FEATURE_TIER"] = feature_tier
        import torch
        self.torch = torch

        code_dir_s = str(code_dir)
        saved = {n: sys.modules.pop(n) for n in POLICY_MODULES if n in sys.modules}
        sys.path.insert(0, code_dir_s)
        try:
            mods = {n: importlib.import_module(n) for n in POLICY_MODULES}
            for n, m in mods.items():
                f = str(Path(getattr(m, "__file__", "")).resolve())
                if not f.startswith(code_dir_s):
                    sys.exit(f"[policy] module '{n}' was imported from {f}, not {code_dir_s}.")
            # Load while the policy modules are registered, in case the
            # checkpoint pickles objects from them.
            ckpt = torch.load(str(policy_path), map_location="cpu", weights_only=False)
        finally:
            sys.path.remove(code_dir_s)
            for n in saved:
                sys.modules[n] = saved[n]

        self.cfg = mods["config"]
        self.features = mods["features"]
        self.models = mods["models"]
        self.arm_mod = mods["arm_state"]
        self.allow_silent = allow_silent
        self.model, self.sat_mean, self.sat_std = self._build_model(ckpt)
        self.model.to(self.cfg.device).eval()

        print(f"[policy] loaded {policy_path.name} | FEATURE_MODE={self.cfg.FEATURE_MODE} "
              f"FEATURE_TIER={self.cfg.FEATURE_TIER} D={self.cfg.D} "
              f"pooled={self.cfg.USE_POOLED_HISTORY} attention={self.cfg.USE_ATTENTION} "
              f"sat_mean={self.sat_mean:.4f} sat_std={self.sat_std:.4f}")
        if self.cfg.FEATURE_MODE != "embeddings_only":
            print("[policy] NOTE: numeric features are active. Two inputs can only be approximated "
                  "here: historical 'which director did the Builder follow' tags (not in CRAFT, "
                  "set to 0) and the current turn's after-move board (unknown at decision time, "
                  "the before-move board is used). Check the divergence-reproduction rate.")

    # -- loading ------------------------------------------------------------
    def _build_model(self, ckpt):
        torch = self.torch
        nn = torch.nn
        if isinstance(ckpt, nn.Module):
            return ckpt, 0.0, 1.0

        if not isinstance(ckpt, dict):
            sys.exit(f"[policy] Unrecognized checkpoint type {type(ckpt).__name__}.")

        def is_state(d):
            return isinstance(d, dict) and d and all(hasattr(v, "shape") for v in d.values())

        state = None
        for k in ("model_state", "state_dict", "model_state_dict", "model"):
            v = ckpt.get(k)
            if isinstance(v, nn.Module):
                return v, *self._sat_stats(ckpt)
            if is_state(v):
                state = v
                break
        if state is None and is_state(ckpt):
            state = ckpt
        if state is None:
            sys.exit(f"[policy] No state dict found in checkpoint. Top-level keys: {sorted(ckpt.keys())}")

        self._warn_config_mismatch(ckpt)

        first_w = next((v for v in state.values() if getattr(v, "ndim", 0) == 2), None)
        if first_w is not None and not self.cfg.USE_ATTENTION and first_w.shape[1] != self.cfg.D:
            sys.exit(f"[policy] Checkpoint expects input dim {first_w.shape[1]} but config.D="
                     f"{self.cfg.D}. Set USE_POOLED_HISTORY in config.py to match this policy.")

        hidden_opts = [self.cfg.POLICY_HIDDEN_DIM]
        if first_w is not None and first_w.shape[0] not in hidden_opts:
            hidden_opts.append(int(first_w.shape[0]))
        errors = []
        for h in hidden_opts:
            try:
                model = self.models.ArmStateEvaluator(input_dim=self.cfg.D, hidden_dim=h,
                                                      use_attention=self.cfg.USE_ATTENTION)
                model.load_state_dict(state)
                return model, *self._sat_stats(ckpt)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"hidden_dim={h}: {exc}")
        sys.exit("[policy] Could not load weights into ArmStateEvaluator:\n  " + "\n  ".join(errors))

    @staticmethod
    def _sat_stats(ckpt) -> Tuple[float, float]:
        src = ckpt.get("extra") if isinstance(ckpt.get("extra"), dict) else ckpt
        return float(src.get("sat_mean", 0.0)), float(src.get("sat_std", 1.0))

    def _warn_config_mismatch(self, ckpt) -> None:
        stored = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else ckpt
        for key in ("FEATURE_MODE", "FEATURE_TIER", "USE_POOLED_HISTORY", "USE_ATTENTION"):
            if key in stored and stored[key] != getattr(self.cfg, key):
                print(f"[policy] WARNING: checkpoint has {key}={stored[key]!r} but config has "
                      f"{getattr(self.cfg, key)!r}. Set it to match (env var or config.py).")

    # -- arm states ---------------------------------------------------------
    def make_arm(self, d, utterance, t, structure_id, combo, before, after,
                 tag=0, sat=0.0, move_failed=""):
        A = self.arm_mod
        arm = A.ArmState(
            speaker=f"{d}_message", structure=structure_id, modelCombo=combo,
            utterance=utterance or "", timestamp=t,
            constraint=A.ConstraintBelief(tag), satisfaction=A.Satisfaction(sat),
            structureBefore=copy.deepcopy(before), structureAfter=copy.deepcopy(after),
        )
        arm.moveFailed = move_failed
        return arm

    def craft_histories(self, craft_index, structure_id, director_model, upto_turn,
                        combo, sat_fn, after_fn) -> Dict[str, List]:
        """Each director's ArmStates for CRAFT turns < upto_turn (turn 0 is
        dropped by encode_arm_state, as in training)."""
        hist = {d: [] for d in DIRECTORS}
        for s in range(0, upto_turn):
            r = craft_index.get((s, structure_id, director_model))
            if r is None:
                continue
            self.append_craft_turn(hist, r, s, structure_id, combo, sat_fn, after_fn(s))
        return hist

    def append_craft_turn(self, hist, row, t, structure_id, combo, sat_fn, after,
                          chosen=None, move_failed=None):
        before = parse_board(row.get("structure_before")) if row.get("structure_before") else None
        failed = move_failed if move_failed is not None else (
            "True" if str(row.get("failed_move", "")).lower() in ("true", "1") else "")
        board_for_sat = after if after is not None else before
        for d in DIRECTORS:
            tag = 0 if chosen is None else (1 if d == chosen else -1)
            hist[d].append(self.make_arm(
                d, row.get(f"{d}_message") or "", t, structure_id, combo, before, after,
                tag=tag, sat=sat_fn(d, board_for_sat), move_failed=failed))

    def _trim(self, arms: List) -> List:
        real = [a for a in arms if a.timestamp != 0]
        keep = self.cfg.HISTORY_LEN + 1
        return real[-keep:] if len(real) > keep else real

    # -- scoring ------------------------------------------------------------
    def score(self, histories: Dict[str, List]) -> Dict[str, float]:
        torch = self.torch
        xs, rs = [], []
        for d in DIRECTORS:
            x, r = self.features.encode_arm_state(self._trim(histories[d]), self.sat_mean, self.sat_std)
            xs.append(x)
            rs.append(r)
        dev = self.cfg.device
        with torch.no_grad():
            out = self.model(torch.stack(xs).to(dev), torch.stack(rs).to(dev))
        return dict(zip(DIRECTORS, out.view(-1).cpu().tolist()))

    def choose(self, scores: Dict[str, float], speaking: List[str]) -> Tuple[str, str]:
        raw = max(DIRECTORS, key=lambda d: scores[d])
        if self.allow_silent or raw in speaking or not speaking:
            return raw, raw
        return max(speaking, key=lambda d: scores[d]), raw


def make_sat_fn(target_raw):
    """Satisfaction as defined in arm_state.py: IoU of the board projected onto
    a director's partial view vs that director's view of the target. Only
    affects the policy under a privileged FEATURE_TIER."""
    def sat(d, board):
        if board is None:
            return 0.0
        try:
            return calculate_iou_board(normalize_structure(getParticalView(d, board)),
                                       normalize_structure(getParticalView(d, target_raw)))
        except Exception:  # noqa: BLE001
            return 0.0
    return sat


# --------------------------------------------------------------------------- #
# Random single-director baseline
# --------------------------------------------------------------------------- #
RANDOM_STEP0_MODES = ("nonhistory", "random", "policy")
ARMS = ("counterfactual", "factual", "random")
ARM_PREFIX = {"counterfactual": "cf", "factual": "fa", "random": "ra"}
# Paired comparisons reported in the summary: (minuend, subtrahend).
ARM_PAIRS = (("counterfactual", "factual"), ("random", "factual"), ("counterfactual", "random"))


def random_draw(seed: int, *identity, pool: List[str]) -> str:
    """
    Uniform draw from `pool`, seeded from --random-seed plus the identity of
    the decision (game, turn, arm, step). A string seed makes random.Random
    deterministic across processes, so draws are reproducible, independent of
    the order turns are run in, and identical on --resume.
    """
    key = "|".join(str(x) for x in (seed,) + identity)
    return random.Random(key).choice(sorted(pool))


# --------------------------------------------------------------------------- #
# Follow-on director turns
# --------------------------------------------------------------------------- #
FOLLOWON_SOURCES = ("generate", "history")
FOLLOWON_SELECTS = ("all", "best", "policy", "recommend")
POLICY_SELECTS = ("policy", "recommend")
# Selection rules after which the factual arm defaults to 'all'.
FACTUAL_ALL_BY_DEFAULT = POLICY_SELECTS
DEFAULT_RECOMMEND_TEMPLATE = ("RECOMMENDATION: Prioritize {director}'s instruction. "
                              "Use the other directors' messages to resolve ambiguity.")
CONVERSATION_KEYS = ("conversation_snapshot", "conversation_history", "conversation")


def load_director_module(module_name: str):
    """Import the module that defines DirectorAgent (and get_director_views_fn)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"[directors] Could not import '{module_name}' ({exc}).\n"
            f"            Pass --director-module <name> (the .py file that defines DirectorAgent, "
            f"without '.py') and run from the folder that also has structure_generator_v2.py "
            f"and task_progress_tracker.py."
        )
    if not hasattr(module, "DirectorAgent"):
        sys.exit(f"[directors] Module '{module_name}' has no DirectorAgent class.")
    return module


def make_local_director_class(DirectorAgent):
    class LocalDirectorAgent(DirectorAgent):
        """
        DirectorAgent served by a ChatBackend (Ollama or the Claude API). The
        prompt (create_enhanced_director_prompt_with_references) and the parser
        (parse_director_response) are inherited unchanged; only the transport
        differs.

        DirectorAgent.__init__ is deliberately not called: it needs torch, it
        builds a hosted-API client, and it re-derives the archetype from
        hash(). Here the archetype comes from the CRAFT row (D*_archetype) so
        the generated directors keep the personalities they had in the
        historical game.
        """

        def __init__(self, director_id, archetype, backend, model_name,
                     temperature=0.7, max_tokens=500, seed=None, fallback_key=""):
            self.director_id = director_id
            self.archetype_source = "craft"
            if archetype not in DirectorAgent.ARCHETYPES:
                archetype = random.Random(f"{fallback_key}-{director_id}").choice(DirectorAgent.TYPES)
                self.archetype_source = "fallback"
            self.archetype = archetype
            self.personality = DirectorAgent.ARCHETYPES[archetype]
            self.use_api = True
            self.backend = backend
            self.provider = backend.name
            self.local_model = None
            self.local_tokenizer = None
            self.client = None  # all generation goes through self.backend
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.seed = seed
            self.last_prompt = ""
            self.last_raw = ""

        def build_prompt(self, current_view, target_view, conversation_history):
            return self.create_enhanced_director_prompt_with_references(
                current_view, target_view, conversation_history, AVAILABLE_BLOCKS
            )

        def generate_response(self, current_view, target_view, conversation_history, available_blocks):
            prompt = self.create_enhanced_director_prompt_with_references(
                current_view, target_view, conversation_history, available_blocks
            )
            self.last_prompt = prompt
            # same system message as DirectorAgent._generate_with_openai
            system = f"You are Director {self.director_id} in a collaborative LEGO construction task."
            try:
                raw = self.backend.complete(
                    self.model_name, system, [{"role": "user", "content": prompt}],
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    seed=self.seed, role="director",
                )
            except Exception as exc:  # noqa: BLE001
                self.last_raw = ""
                return {"internal_thinking": f"Error in generation: {exc}",
                        "public_message": "No message provided", "raw_response": ""}
            self.last_raw = raw
            # parse_director_response prints a lot of debug output; keep the log readable
            with contextlib.redirect_stdout(io.StringIO()):
                return self.parse_director_response(raw)

    return LocalDirectorAgent


def get_views_fn(director_module):
    fn = getattr(director_module, "get_director_views_fn", None)
    if fn is None:
        try:
            from structure_generator_v2 import get_director_views as fn  # type: ignore
        except ImportError:
            fn = None
    return fn


def compute_target_views(views_fn, struct_entry: Dict) -> Dict:
    """Per-director target views in the row_0/row_1/row_2 format the director prompt expects."""
    for k in ("director_views", "target_views", "views"):
        v = struct_entry.get(k)
        if isinstance(v, dict) and all(d in v for d in DIRECTORS):
            return v
    if views_fn is None:
        raise RuntimeError("no get_director_views function available")
    views = views_fn(struct_entry["structure"])
    if isinstance(views, (list, tuple)) and len(views) == 3:
        views = dict(zip(DIRECTORS, views))
    if not (isinstance(views, dict) and all(d in views for d in DIRECTORS)):
        raise RuntimeError(f"get_director_views returned {type(views).__name__}, expected dict keyed D1/D2/D3")
    return views


def is_silent_text(text: Optional[str]) -> bool:
    return (text or "").strip().lower() in SILENT_MARKERS


def historical_director_messages(row: Dict) -> Dict[str, str]:
    out = {}
    for d in DIRECTORS:
        msg = row.get(f"{d}_message")
        out[d] = "" if msg is None else str(msg).strip()
    return out


def snapshot_to_text(value) -> str:
    """CRAFT conversation_snapshot -> plain 'speaker: text' lines."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in "[{":
            try:
                return snapshot_to_text(json.loads(s))
            except json.JSONDecodeError:
                pass
        return s
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                speaker = (item.get("speaker") or item.get("role") or item.get("director")
                           or item.get("name") or "?")
                text = (item.get("message") or item.get("content") or item.get("utterance")
                        or item.get("text") or json.dumps(item))
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(value)


_SNAPSHOT_WARNED = False


def base_conversation(row: Dict, conv_field: Optional[str]) -> str:
    """
    Conversation up to and including the directors' turn-t messages (before the
    Builder's turn-t response, which each arm supplies itself).
    """
    global _SNAPSHOT_WARNED
    convo = snapshot_to_text(row.get(conv_field)) if conv_field else ""
    lines = [convo] if convo else []
    already_included = False
    for d, text in historical_director_messages(row).items():
        if is_silent_text(text):
            continue
        if convo and text[:60] in convo:
            already_included = True
            continue
        lines.append(f"{d}: {text}")
    if already_included and convo:
        # FIX: CRAFT's conversation_snapshot for turn t ends with the HISTORICAL
        # Builder's turn-t reply (including a "Current board" that matches
        # neither arm). Each arm appends its own turn-t Builder line, so keeping
        # the historical one gave follow-on directors two contradictory replies
        # and a wrong board. Cut the snapshot after the last line holding one
        # of this turn's director messages.
        cut = -1
        for d, text in historical_director_messages(row).items():
            if is_silent_text(text):
                continue
            i = convo.rfind(text[:60])
            if i >= 0:
                end = convo.find("\n", i)
                cut = max(cut, len(convo) if end < 0 else end)
        if 0 <= cut < len(convo):
            dropped = convo[cut:].strip()
            convo = convo[:cut]
            lines[0] = convo
            if not _SNAPSHOT_WARNED:
                _SNAPSHOT_WARNED = True
                print(f"[fix] '{conv_field}' contained the divergence turn's historical Builder reply; "
                      f"dropped {len(dropped)} chars after the turn's director messages, e.g.: {dropped[:120]!r}")
    return "\n".join(lines)


def builder_line(move: Dict, executed: bool) -> str:
    action = move.get("action")
    note = (move.get("confirmation") or move.get("interpretation") or "").strip()
    if action == "clarify":
        return f"Builder: {move.get('clarification', '')}".strip()
    if action == "place":
        desc = f"placed {move.get('block')} at {move.get('position')} on layer {move.get('layer')}"
    else:
        desc = f"removed the block at {move.get('position')} on layer {move.get('layer')}"
    if move.get("span_to"):
        desc += f" spanning to {move.get('span_to')}"
    status = "" if executed else " -- MOVE FAILED, the board did not change"
    return f"Builder: {desc}{status}." + (f" {note}" if note else "")


def tail(text: str, max_chars: Optional[int]) -> str:
    if max_chars and len(text) > max_chars:
        return "...\n" + text[-max_chars:]
    return text


_TAG_RESIDUE = re.compile(r"^\s*[^\w\s]{0,2}\s*/?message\s*>?\s*", re.IGNORECASE)
_FAKE_TURN = re.compile(r"^\s*(?:Builder|D[123])\s*:", re.MULTILINE)


def clean_director_text(text: str) -> str:
    """FIX: the local director often leaves '<message>' residue ("$message>",
    "=message>", ":message>") and keeps writing the dialogue: invented Builder
    replies with fake board states, or other directors' lines. Those were
    appended to the shared conversation and seen by later directors and by the
    policy. Keep only this director's own message."""
    t = _TAG_RESIDUE.sub("", text or "", count=1)
    t = re.sub(r"</?message>", "", t, flags=re.IGNORECASE)
    m = _FAKE_TURN.search(t)
    if m:
        t = t[:m.start()]
    return t.strip()


def generate_director_messages(directors: Dict, board, views: Dict, conversation: str,
                               max_history_chars: Optional[int]) -> Dict[str, Dict]:
    """
    New D1/D2/D3 utterances for this arm's board. Directors speak in order and
    each sees the ones who spoke before it this turn, matching the prompt's
    'check if other directors already covered what you need'.
    """
    out: Dict[str, Dict] = {}
    running = conversation
    for d in DIRECTORS:
        parsed = directors[d].generate_response(
            copy.deepcopy(board), views[d], tail(running, max_history_chars), AVAILABLE_BLOCKS
        )
        raw_text = (parsed.get("public_message") or "").strip()
        text = clean_director_text(raw_text)
        out[d] = {
            "text": text,
            "text_raw": raw_text if raw_text != text else None,
            "thinking": parsed.get("internal_thinking"),
            "source": "generated",
            "archetype": directors[d].archetype,
        }
        print(f"      [gen {d} ({directors[d].archetype})] {text[:160]}")
        if not is_silent_text(text):
            running = f"{running}\n{d}: {text}" if running else f"{d}: {text}"
    return out


def run_followon_step(agent, arm, k, msgs, speaking, board, spans, target_norm, select,
                      oracle_moves=None, random_identity=None, random_note_template=None):
    """Builder turn at a follow-on step under the chosen selection rule."""
    if select == "all":
        utterance = "\n".join(f"{d}: {msgs[d]['text']}" for d in speaking)
        step, board, spans = execute_step(agent, arm, k, None, utterance, board, spans, target_norm,
                                          oracle_moves)
        step["selection"] = {"mode": "all", "speaking": speaking}
        return step, board, spans

    if select == "random":
        # Random control arm: the director is drawn uniformly from those who
        # spoke instead of chosen by the policy. Without a note template the
        # Builder hears only that director (control for 'policy'); with one it
        # hears everyone plus a recommendation naming the random director
        # (control for 'recommend').
        chosen = random_draw(*random_identity, pool=speaking)
        print(f"    [{arm:<14} t+{k}] random pick from {speaking} -> {chosen}")
        note = None
        if random_note_template:
            utterance = "\n".join(f"{d}: {msgs[d]['text']}" for d in speaking)
            note = random_note_template.format(director=chosen)
        else:
            utterance = f"{chosen}: {msgs[chosen]['text']}"
        step, board, spans = execute_step(
            agent, arm, k, chosen, utterance,
            board, spans, target_norm, oracle_moves, builder_note=note,
        )
        step["selection"] = {"mode": "random", "chosen": chosen, "speaking": speaking,
                             "as_recommendation": bool(random_note_template),
                             "p_chosen": round(1.0 / len(speaking), 6)}
        return step, board, spans

    # select == "best": the Builder hears each director alone; keep the move
    # with the highest whole-board IoU vs target. Uses the target, so this is
    # an oracle selection rule, not something a deployed policy could do.
    cands = []
    for d in speaking:
        res, b, s = execute_step(
            agent, f"{arm}/{d}", k, d, f"{d}: {msgs[d]['text']}",
            copy.deepcopy(board), copy.deepcopy(spans), target_norm, oracle_moves,
        )
        cands.append((d, res, b, s))
    best_i = max(range(len(cands)), key=lambda i: (cands[i][1]["satisfaction_full"], -i))
    d, step, board, spans = cands[best_i]
    top = step["satisfaction_full"]
    step["selection"] = {
        "mode": "best",
        "chosen": d,
        "speaking": speaking,
        "candidate_satisfaction": {c[0]: c[1]["satisfaction_full"] for c in cands},
        "candidate_moves": {c[0]: c[1]["builder_move"] for c in cands},
        "n_tied_best": sum(1 for c in cands if c[1]["satisfaction_full"] == top),
    }
    print(f"    [{arm:<14} t+{k}] best-of-{len(cands)} chose {d} (sat={top:.4f})")
    return step, board, spans


def run_rollout(
    agent: LocalBuilderAgent,
    arm: str,
    director: str,
    message_obj: Dict,
    row: Dict,
    target_norm: Dict,
    oracle_moves,
    craft_index: Dict,
    turn_index: int,
    structure_id,
    director_model: str,
    horizon: int,
    after_key: Optional[str],
    followon: Dict,
    check_tokens: bool = False,
) -> Dict:
    """
    Step 0: this arm's director utterance on the shared pre-move board.
    Steps 1..H-1 depend on followon['source'] / followon['select']:
      generate: new D1-D3 utterances from the DirectorAgent prompt, conditioned
                on THIS arm's board and conversation (continuous causal chain)
      history:  the logged CRAFT D1-D3 messages for turn t+k
      all:      Builder hears every non-silent director
      best:     Builder hears each director alone; best-scoring move is kept
      random:   Builder hears one speaking director drawn uniformly at random
    Every step is scored against the target and the historical board.
    """
    source, select = followon["source"], followon["select"]
    board = copy.deepcopy(parse_board(message_obj["structureBefore"]))
    spans = copy.deepcopy(json.loads(row["spans_before"]))
    steps: List[Dict] = []
    stop_reason = None
    conversation = base_conversation(row, followon["conversation_field"])
    directors = followon["make_directors"](row) if (source == "generate" and horizon > 1) else None
    policy = followon.get("policy") if select in POLICY_SELECTS else None
    combo = message_obj.get("modelCombo")
    sat_fn = followon.get("sat_fn")
    hist = None
    if policy is not None:
        hist = policy.craft_histories(
            craft_index, structure_id, director_model, turn_index, combo, sat_fn,
            lambda s_: ground_truth_board_after(craft_index, s_, structure_id, director_model, after_key),
        )

    for k in range(horizon):
        t = turn_index + k
        if k == 0:
            note0 = None
            if followon.get("step0_mode") == "recommend":
                # Both arms hear all three historical messages; only the
                # recommended director (this arm's step-0 director) differs.
                hist_msgs = historical_director_messages(row)
                utterance = "\n".join(f"{d}: {txt}" for d, txt in hist_msgs.items()
                                      if not is_silent_text(txt))
                note0 = followon["recommend_template"].format(director=director)
            else:
                utterance = f"{director}: {message_obj['utterance']}"
            board_pre = copy.deepcopy(board)
            step, board, spans = execute_step(
                agent, arm, 0, director, utterance, board, spans,
                target_norm, oracle_moves, check_tokens=check_tokens, builder_note=note0,
            )
            step["selection"] = {"mode": "divergence", "chosen": director,
                                 "step0_mode": followon.get("step0_mode") or "single",
                                 "source": followon.get("step0_source", {}).get(arm, "logged")}
            step["oracle_source"] = "craft" if oracle_moves else None
            if policy is not None:
                row_t = dict(row)
                row_t["structure_before"] = board_pre
                policy.append_craft_turn(
                    hist, row_t, t, structure_id, combo, sat_fn, copy.deepcopy(board),
                    chosen=director, move_failed="" if step["move_executed"] else "True")
            new_lines = []
        else:
            if source == "history":
                frow = craft_index.get((t, structure_id, director_model))
                if frow is None:
                    stop_reason = f"game ended: no CRAFT row for turn {t}"
                    break
                msgs = {d: {"text": txt, "source": "history"}
                        for d, txt in historical_director_messages(frow).items()}
            else:
                if steps and steps[-1]["satisfaction_full"] >= 1.0:
                    stop_reason = "target complete"
                    break
                msgs = generate_director_messages(
                    directors, board, followon["target_views"], conversation,
                    followon["max_history_chars"],
                )
            speaking = [d for d in DIRECTORS if not is_silent_text(msgs[d]["text"])]
            if not speaking:
                stop_reason = f"all directors silent at turn {t}"
                break
            step_oracle, oracle_meta, oracle_flagged = None, None, None
            if followon.get("oracle") is not None:
                step_oracle, all_cands = followon["oracle"].candidates(
                    board, spans, followon["target_raw"], followon["target_spans"])
                flagged = defaultdict(int)
                for c in all_cands:
                    if c.get("flag", "ok") != "ok":
                        flagged[c.get("flag")] += 1
                oracle_meta = {"source": "regenerated", "n_usable": len(step_oracle),
                               "flagged": dict(flagged)}
                oracle_flagged = [c for c in all_cands if c.get("flag", "ok") != "ok"]
                print(f"    [{arm:<14} t+{k}] oracle regenerated: {len(step_oracle)} usable"
                      + (f", flagged {dict(flagged)}" if flagged else ""))
            if policy is not None:
                board_pre = copy.deepcopy(board)
                cands = {
                    d: policy.make_arm(d, msgs[d]["text"], t, structure_id, combo,
                                       board_pre, board_pre, tag=0, sat=sat_fn(d, board_pre))
                    for d in DIRECTORS
                }
                scores = policy.score({d: hist[d] + [cands[d]] for d in DIRECTORS})
                chosen, raw = policy.choose(scores, speaking)
                print(f"    [{arm:<14} t+{k}] policy scores "
                      + " ".join(f"{d}={scores[d]:+.4f}" for d in DIRECTORS)
                      + f" -> {chosen}" + (f" (raw argmax {raw} is silent)" if raw != chosen else ""))
                note = None
                if select == "recommend":
                    # Builder hears every speaking director; the policy's pick
                    # is a recommendation appended to the prompt.
                    utterance = "\n".join(f"{d}: {msgs[d]['text']}" for d in speaking)
                    note = followon["recommend_template"].format(director=chosen)
                else:
                    utterance = f"{chosen}: {msgs[chosen]['text']}"
                step, board, spans = execute_step(
                    agent, arm, k, chosen, utterance,
                    board, spans, target_norm, step_oracle, builder_note=note,
                )
                step["selection"] = {
                    "mode": select, "chosen": chosen, "raw_argmax": raw,
                    "picked_silent": raw != chosen, "scores": scores, "speaking": speaking,
                }
                for d in DIRECTORS:
                    cands[d].constraint = policy.arm_mod.ConstraintBelief(1 if d == chosen else -1)
                    cands[d].structureAfter = copy.deepcopy(board)
                    cands[d].moveFailed = "" if step["move_executed"] else "True"
                    hist[d].append(cands[d])
            else:
                step, board, spans = run_followon_step(
                    agent, arm, k, msgs, speaking, board, spans, target_norm, select,
                    oracle_moves=step_oracle,
                    random_identity=(followon.get("random_seed", 0), "followon",
                                     structure_id, turn_index, combo, arm, k),
                    random_note_template=(followon["recommend_template"]
                                          if followon.get("random_as_recommend") else None),
                )
            step["director_messages"] = msgs
            step["oracle_source"] = "regenerated" if oracle_meta else None
            step["oracle_regeneration"] = oracle_meta
            step["oracle_flagged"] = copy.deepcopy(oracle_flagged)
            new_lines = [f"{d}: {msgs[d]['text']}" for d in speaking]

        step["turn"] = t
        new_lines.append(builder_line(step["builder_move"], step["move_executed"]))
        conversation = "\n".join([c for c in [conversation] + new_lines if c])

        gt_after = ground_truth_board_after(craft_index, t, structure_id, director_model, after_key)
        if gt_after is not None:
            gt_norm = normalize_structure(gt_after)
            gt_sat = calculate_iou_board(gt_norm, target_norm)
            step["ground_truth_satisfaction_full"] = gt_sat
            step["iou_vs_ground_truth_board"] = calculate_iou_board(normalize_structure(board), gt_norm)
            step["progress_vs_ground_truth"] = step["satisfaction_full"] - gt_sat
        else:
            step["ground_truth_satisfaction_full"] = None
            step["iou_vs_ground_truth_board"] = None
            step["progress_vs_ground_truth"] = None

        print(
            f"    [{arm:<14} t+{k}] sat={step['satisfaction_full']:.4f}  "
            f"gt_sat={_fmt(step['ground_truth_satisfaction_full'])}  "
            f"iou_vs_gt={_fmt(step['iou_vs_ground_truth_board'])}"
        )
        steps.append(step)

    return {
        "arm": arm,
        "director": director,
        "silent_director": is_silent(message_obj),
        "followon_source": source,
        "followon_select": select,
        "horizon_requested": horizon,
        "horizon_reached": len(steps),
        "stop_reason": stop_reason,
        "steps": steps,
        "final_conversation": conversation,
    }


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def mean_sem(values: List[float]):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return mean, sem, n


def pack(values: List[float]) -> Dict:
    mean, sem, n = mean_sem(values)
    return {
        "n": n,
        "mean": None if mean is None else round(mean, 6),
        "sem": None if sem is None else round(sem, 6),
        "n_positive": sum(1 for v in values if v > 0),
        "n_zero": sum(1 for v in values if v == 0),
        "n_negative": sum(1 for v in values if v < 0),
    }


def print_table(title: str, endpoints: Dict[str, Dict], indent: str = "  ") -> None:
    print(f"{indent}{title}")
    print(f"{indent}  {'endpoint':<40}{'n':>5}{'mean':>12}{'sem':>11}{'+/0/-':>12}")
    for name, block in endpoints.items():
        sem = "n/a" if block["sem"] is None else f"{block['sem']:.5f}"
        mean = "n/a" if block["mean"] is None else f"{block['mean']:+.5f}"
        split = f"{block['n_positive']}/{block['n_zero']}/{block['n_negative']}"
        print(f"{indent}  {name:<40}{block['n']:>5}{mean:>12}{sem:>11}{split:>12}")


def estimate_max_calls(n_turns: int, n_arms: int, horizon: int, source: str,
                       selects: Dict[str, str], max_retries: int) -> Tuple[int, int]:
    """Upper bound on (builder, director) model calls, ignoring early stops.
    Builder retries on unparseable output are counted in the builder bound."""
    builder = director = 0
    for select in selects.values():
        per_followon_builder = 3 if select == "best" else 1
        builder += 1 + (horizon - 1) * per_followon_builder
        if source == "generate":
            director += (horizon - 1) * 3
    scale = n_turns * (n_arms / max(len(selects), 1))
    return int(builder * scale * (1 + max_retries)), int(director * scale)


# --------------------------------------------------------------------------- #
# Incremental saving and --resume
# --------------------------------------------------------------------------- #
RUN_CONFIG_NAME = "_run_config.json"
# Settings that do not change results, so they may differ when resuming.
RESUME_VOLATILE_ARGS = {
    "limit", "resume", "overwrite", "print_prompts", "save_prompts", "out_dir", "out_root",
    "anthropic_api_key", "skip_preflight", "api_max_retries", "min_request_interval",
}


def _json_default(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else str(obj)


def write_json_atomic(path: Path, obj) -> None:
    """Write to a temp file, then swap it in: an interrupted write can never
    leave a truncated results file behind."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def serializable_args(args) -> Dict:
    raw = {k: (str(v) if isinstance(v, Path) else v)
           for k, v in sorted(vars(args).items()) if k != "anthropic_api_key"}
    return json.loads(json.dumps(raw, default=str))


def record_identity(r: Dict) -> tuple:
    """One divergence turn: same game, turn, and the two directors being compared."""
    return (r.get("turn"), str(r.get("structure_id")), r.get("modelCombo"),
            r.get("counterfactualDirector"), r.get("historicalDirector"))


def game_files(out_dir: Path) -> List[Path]:
    return [p for p in sorted(out_dir.glob("*.json")) if not p.name.startswith("_")]


def load_saved_records(out_dir: Path) -> Dict[str, List[Dict]]:
    saved: Dict[str, List[Dict]] = {}
    for p in game_files(out_dir):
        try:
            with open(p, encoding="utf-8") as fh:
                rows = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.exit(f"[resume] Could not read {p} ({exc}). Move it aside and resume again.")
        saved[p.name[:-len(".json")]] = rows if isinstance(rows, list) else [rows]
    return saved


def saved_run_args(out_dir: Path) -> Optional[Dict]:
    cfg = out_dir / RUN_CONFIG_NAME
    candidates = [cfg] if cfg.exists() else sorted(out_dir.glob("_run_summary*.json"))
    for p in candidates:
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh).get("run_args")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def resume_conflicts(prev: Dict, cur: Dict) -> Dict:
    """Result-affecting settings that differ. Keys only one side knows (older
    script versions) are ignored."""
    keys = (set(prev) & set(cur)) - RESUME_VOLATILE_ARGS
    return {k: (prev[k], cur[k]) for k in sorted(keys) if prev[k] != cur[k]}


# --------------------------------------------------------------------------- #
# Automatic output folder naming
# --------------------------------------------------------------------------- #
def _slug(value) -> str:
    """Filesystem-safe model/policy name: drops ':latest', ':' -> '-'."""
    s = re.sub(r":latest$", "", str(value))
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def divergence_tag(div_dir: Path) -> Tuple[str, str]:
    """('policy_seed77_b0.9_...', 'validation') from .../divergenceData_policy_seed77_.../validation."""
    split = div_dir.name
    parent = div_dir.parent.name
    prefix = "divergenceData_"
    tag = parent[len(prefix):] if parent.startswith(prefix) else parent
    return (tag or "divergence"), split


def auto_out_dir(args, cf_select: str, fa_select: str, director_model_name: str) -> Path:
    """
    Folder name built from the settings that change results, joined by '__':
      <local|claude>_<builder>__<policy>__<split>__<filter>__<oracle>__h<H>_n<N>__<source>-<select>
    plus optional parts only when they differ from the defaults:
      fa-<select>      factual follow-on rule, if not the default
      noRand           random control arm switched off
      r0<mode>         random arm's step-0 rule, if not 'nonhistory'
      rseed<k>         random arm's seed, if not 0
      dir-<model>      generated-director model, if different from the Builder
      eff-<level>      Claude effort level, if set
      fpol-<policy>    follow-on policy .pt, if not the policy behind the divergence data
      skipSilent / seed<k> / shuf<k> / noshuffle
    """
    tag, split = divergence_tag(args.divergence_dir)
    oracle = ("oracleEvery" if args.oracle_every_turn
              else "oracle0" if args.use_oracle else "noOracle")
    backend_prefix = "local" if args.backend == "ollama" else "claude"
    parts = [
        f"{backend_prefix}_{_slug(args.model)}",
        tag,
        split,
        "decidable" if args.decidability_file else "unfiltered",
        oracle,
        f"h{args.horizon}_n{args.limit if args.limit else 'all'}",
    ]
    if args.horizon > 1:
        parts.append(f"{args.followon_source}-{cf_select}")
        if cf_select == "recommend" and getattr(args, "recommend_step0", "recommend") == "single":
            parts.append("s0single")
        default_fa = "all" if cf_select in FACTUAL_ALL_BY_DEFAULT else cf_select
        if fa_select != default_fa:
            parts.append(f"fa-{fa_select}")
        if args.followon_source == "generate" and director_model_name != args.model:
            parts.append(f"dir-{_slug(director_model_name)}")
    if args.no_random_arm:
        parts.append("noRand")
    else:
        if args.random_step0 != "nonhistory":
            parts.append(f"r0{args.random_step0}")
        if args.random_seed != 0:
            parts.append(f"rseed{args.random_seed}")
    if args.effort and "anthropic" in (args.backend, args.director_backend):
        parts.append(f"eff-{args.effort}")
    if args.policy_path and args.policy_path.stem != tag:
        parts.append(f"fpol-{_slug(args.policy_path.stem)}")
    if args.skip_silent:
        parts.append("skipSilent")
    if args.seed != 0:
        parts.append(f"seed{args.seed}")
    if args.no_shuffle:
        parts.append("noshuffle")
    elif args.shuffle_seed != 42:
        parts.append(f"shuf{args.shuffle_seed}")

    root = args.out_root or (args.divergence_dir.parent.parent / "multiTurn")
    return root / "__".join(parts)


def unique_out_dir(path: Path, overwrite: bool) -> Path:
    """Never write into a folder that already holds results unless asked to."""
    def has_results(p: Path) -> bool:
        return p.exists() and any(p.iterdir())
    if overwrite or not has_results(path):
        return path
    i = 2
    while has_results(path.with_name(f"{path.name}__run{i}")):
        i += 1
    return path.with_name(f"{path.name}__run{i}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired factual/counterfactual multi-turn Builder rollouts with a local Ollama "
                    "model or a Claude model over the Anthropic API."
    )
    ap.add_argument("--divergence-dir", required=True, type=Path)
    ap.add_argument("--structures", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output folder. Default: named automatically from the run settings "
                         "under --out-root (see auto_out_dir).")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Parent folder for auto-named outputs. Default: 'multiTurn' next to the "
                         "divergenceData_* folder.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reuse the output folder even if it already has results (default: a "
                         "new __runN folder is used instead).")
    ap.add_argument("--backend", choices=BACKENDS, default="ollama",
                    help="Where the Builder (and, by default, the generated directors) run: "
                         "'ollama' (local, default) or 'anthropic' (Claude over the API).")
    ap.add_argument("--director-backend", choices=BACKENDS, default=None,
                    help="Backend for generated directors. Default: same as --backend.")
    ap.add_argument("--model", default=None,
                    help=f"Builder model. Default: {DEFAULT_MODEL} (ollama) or "
                         f"{DEFAULT_CLAUDE_MODEL} (anthropic).")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama endpoint (ollama backend only).")
    ap.add_argument("--anthropic-api-key", default=None,
                    help="Anthropic API key. Default: the ANTHROPIC_API_KEY environment variable.")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                    help="(anthropic) Effort level for models that support it. Lower is faster and "
                         "cheaper. Dropped automatically if the model rejects it.")
    ap.add_argument("--no-send-temperature", action="store_true",
                    help="(anthropic) Don't send --temperature/--director-temperature. Without this "
                         "flag they are sent, and dropped automatically if the model rejects them.")
    ap.add_argument("--api-max-retries", type=int, default=8,
                    help="(anthropic) SDK retries on rate-limit/overload/connection errors.")
    ap.add_argument("--min-request-interval", type=float, default=0.0,
                    help="(anthropic) Minimum seconds between API calls, to stay under rate limits.")
    ap.add_argument("--dataset", default="Abhijnan/craft-benchmark-lean")
    ap.add_argument("--arms", choices=["both", "counterfactual", "factual"], default="both",
                    help="Policy/history arms to run. The random control arm runs in addition "
                         "unless --no-random-arm is given.")
    ap.add_argument("--no-random-arm", action="store_true",
                    help="Skip the random control arm (runs by default; see 'Random control arm').")
    ap.add_argument("--horizon", type=int, default=3,
                    help="Turns to roll out from each divergence point, including the divergence "
                         "turn itself (3 = turns t, t+1, t+2).")
    ap.add_argument("--followon-source", choices=FOLLOWON_SOURCES, default="generate",
                    help="Where follow-on director utterances come from. 'generate' (default): new "
                         "D1-D3 utterances from the DirectorAgent prompt on each arm's own board, a "
                         "continuous causal chain. 'history': replay the logged CRAFT messages.")
    ap.add_argument("--followon-select", choices=FOLLOWON_SELECTS, default="all",
                    help="What the Builder hears at follow-on turns. 'all' (default): every "
                         "non-silent director. 'best': each director alone, keeping the move with "
                         "the highest IoU vs target (oracle selection). history+best ablates "
                         "compounding returns from director selection alone. 'policy': the "
                         "policy's pick only. 'recommend': all directors plus the policy's pick "
                         "as a recommendation in the Builder prompt. (The random control arm "
                         "always mirrors this rule with random picks.)")
    ap.add_argument("--director-module", default="director_agent_local",
                    help="Module (file name without .py) that defines DirectorAgent.")
    ap.add_argument("--director-model", default=None,
                    help="Model for generated directors. Default: same as --model when the "
                         "director backend matches the Builder's, otherwise that backend's default.")
    ap.add_argument("--director-temperature", type=float, default=0.7,
                    help="Matches DirectorAgent's API temperature.")
    ap.add_argument("--director-max-tokens", type=int, default=None,
                    help="Default: 500 (ollama) or 4096 (anthropic, to leave room for thinking).")
    ap.add_argument("--conversation-field", default=None,
                    help="CRAFT column with the dialogue so far, used as director conversation "
                         "history. Auto-detected; pass 'none' to start from the divergence turn only.")
    ap.add_argument("--max-history-chars", type=int, default=6000,
                    help="Keep only the last N characters of conversation in director prompts, so "
                         "the prompt stays inside the Ollama context window. 0 = no cap.")
    ap.add_argument("--policy-path", type=Path, default=None,
                    help="Trained index policy (.pt). Required for --followon-select policy and "
                         "--policy-check-only.")
    ap.add_argument("--policy-code-dir", type=Path, default=None,
                    help="Folder with the policy project's config.py/features.py/models.py/"
                         "arm_state.py. Default: searched next to --policy-path, then cwd.")
    ap.add_argument("--policy-feature-mode", default=None,
                    help="Override FEATURE_MODE for the policy (must match how it was trained).")
    ap.add_argument("--policy-feature-tier", default=None,
                    help="Override FEATURE_TIER for the policy (must match how it was trained).")
    ap.add_argument("--policy-allow-silent", action="store_true",
                    help="Let the policy pick a silent director at follow-on turns. Default: if "
                         "the argmax is silent, take the best-scoring director who spoke (logged).")
    ap.add_argument("--factual-followon-select", choices=FOLLOWON_SELECTS, default=None,
                    help="Follow-on selection for the FACTUAL arm. Default: same as "
                         "--followon-select, except 'all' when that is 'policy' or 'recommend'.")
    ap.add_argument("--recommend-step0", choices=["recommend", "single"], default="recommend",
                    help="(recommend mode) Divergence turn: 'recommend' (default) gives BOTH arms all "
                         "three historical messages plus a recommendation naming that arm's director; "
                         "'single' keeps the original step 0 (each arm hears only its director).")
    ap.add_argument("--recommend-template", default=DEFAULT_RECOMMEND_TEMPLATE,
                    help="(recommend mode) Recommendation appended to the Builder prompt; must "
                         "contain {director}.")
    ap.add_argument("--random-step0", choices=RANDOM_STEP0_MODES, default="nonhistory",
                    help="Random arm's divergence-turn director: 'nonhistory' (default) uniform "
                         "over speaking directors other than history's, mirroring the policy arm; "
                         "'random' uniform over all speaking directors; 'policy' the policy's "
                         "logged pick, so only the follow-on selection rule is randomized.")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="Seed for the random arm's draws. Draws also depend on the decision's "
                         "identity, so they are reproducible and stable under --resume.")
    ap.add_argument("--policy-check-only", action="store_true",
                    help="Rebuild the policy's input at every divergence turn, check its argmax "
                         "matches the logged 'predicted' director, report the rate and exit. "
                         "No model calls.")
    ap.add_argument("--preview-director-prompt", action="store_true",
                    help="Print the D1 prompt for the first usable turn and exit (no model calls).")
    ap.add_argument("--followon-utterance-field", default=None,
                    help="(history source only) CRAFT column with the historical discussion. "
                         "Auto-detected; check with --inspect-row.")
    ap.add_argument("--inspect-row", action="store_true",
                    help="Print one CRAFT row's columns and the auto-detected fields, then exit.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Builder max tokens. Default: 400 (ollama) or 4096 (anthropic).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Ollama sampling seed; -1 disables. Ignored by the Claude API.")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--decidability-file", type=Path, default=None)
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Builder re-prompts on unparseable output.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-silent", action="store_true")
    ap.add_argument("--use-oracle", action="store_true",
                    help="Pass CRAFT oracle_moves at step 0 only (see module docstring).")
    ap.add_argument("--oracle-every-turn", action="store_true",
                    help="Oracle candidates at EVERY turn: step 0 uses CRAFT's logged oracle_moves "
                         "(history); each follow-on turn regenerates them for that arm's own board "
                         "with oracle.py. Implies --use-oracle.")
    ap.add_argument("--save-prompts", choices=["step0", "all", "none"], default="step0",
                    help="Which steps keep the full Builder prompt (builder_prompt / "
                         "builder_system_prompt) in the output files. Default: step0. "
                         "'all' saves every step (larger files). Every step always keeps "
                         "builder_prompt_sha1 and builder_note.")
    ap.add_argument("--print-prompts", type=int, default=0, metavar="N",
                    help="Print the full Builder prompt to the console for every step of the "
                         "first N divergence turns run (both arms). Default 0 (off).")
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted run: reuse its output folder (instead of a new "
                         "__runN), skip turns already saved there, and build the summary from saved "
                         "plus new results. Refuses if result-affecting settings differ.")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    PROMPT_LOG["save"] = args.save_prompts

    if args.horizon < 1:
        sys.exit("--horizon must be >= 1")
    if args.oracle_every_turn:
        args.use_oracle = True

    # Backend-dependent defaults.
    args.director_backend = args.director_backend or args.backend
    args.model = args.model or DEFAULT_MODEL_BY_BACKEND[args.backend]
    if args.director_model is None:
        args.director_model = (args.model if args.director_backend == args.backend
                               else DEFAULT_MODEL_BY_BACKEND[args.director_backend])
    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS[args.backend][0]
    if args.director_max_tokens is None:
        args.director_max_tokens = DEFAULT_MAX_TOKENS[args.director_backend][1]

    backends: Dict[str, ChatBackend] = {}

    def get_backend(name: str) -> ChatBackend:
        if name not in backends:
            if name == "ollama":
                backends[name] = OllamaBackend(base_url=args.base_url)
            else:
                backends[name] = AnthropicBackend(
                    api_key=args.anthropic_api_key,
                    max_retries=args.api_max_retries,
                    send_temperature=not args.no_send_temperature,
                    effort=args.effort,
                    min_interval=args.min_request_interval,
                )
        return backends[name]

    cf_select = args.followon_select
    fa_select = args.factual_followon_select or (
        "all" if cf_select in FACTUAL_ALL_BY_DEFAULT else cf_select)
    if "{director}" not in args.recommend_template:
        sys.exit("--recommend-template must contain {director}.")
    step0_mode = ("recommend" if cf_select == "recommend" and args.recommend_step0 == "recommend"
                  else "single")
    if not args.no_random_arm:
        print(f"[random] Random control arm ON: mirrors the counterfactual arm "
              f"('{cf_select}') with uniformly random director picks; step 0 = "
              f"'{args.random_step0}'; random seed {args.random_seed}.")

    # Output folder: explicit --out-dir wins; otherwise build the name from the
    # settings so runs never collide. Either way, never overwrite silently.
    _dir_model = args.director_model
    out_explicit = args.out_dir is not None
    if not out_explicit:
        args.out_dir = auto_out_dir(args, cf_select, fa_select, _dir_model)
    if not args.resume:
        args.out_dir = unique_out_dir(args.out_dir, args.overwrite)
    writes_output = not (args.policy_check_only or args.preview_director_prompt or args.inspect_row)
    if writes_output:
        print(f"[out] {'Output folder' if out_explicit else 'Auto-named output folder'}: {args.out_dir}")
        print(f"[backend] Builder: {args.backend}/{args.model} (max_tokens={args.max_tokens})")
        if args.followon_source == "generate" and args.horizon > 1:
            print(f"[backend] Directors: {args.director_backend}/{args.director_model} "
                  f"(max_tokens={args.director_max_tokens})")
        if "anthropic" in (args.backend, args.director_backend):
            print("[backend] Claude API has no sampling seed: arms with identical prompts can still "
                  "differ. Compare builder_prompt_sha1 across arms when reading deltas.")

    div_tag, _ = divergence_tag(args.divergence_dir)
    if args.policy_path and args.policy_path.stem != div_tag:
        print(f"[warn] Follow-on policy '{args.policy_path.stem}' is not the policy that produced "
              f"this divergence data ('{div_tag}'). Step 0 uses the logged picks of '{div_tag}', "
              f"follow-on turns would use a different policy, and the reproduction check will "
              f"likely fail. Intended only if you are deliberately mixing policies.")
    needs_policy = args.policy_check_only or any(s in POLICY_SELECTS for s in (cf_select, fa_select))
    if needs_policy and args.policy_path is None:
        sys.exit("--policy-path is required for --followon-select policy / --policy-check-only.")

    craft_index = build_craft_index(args.dataset)
    if args.inspect_row:
        inspect_craft_row(craft_index)
        return

    sample_row = next(iter(craft_index.values()))
    if not has_director_message_columns(sample_row):
        sys.exit("[data] CRAFT rows need D1_message/D2_message/D3_message "
                 "(turn-t conversation and the history follow-on source use them).")
    if args.horizon > 1:
        print(f"[data] Follow-on turns: source={args.followon_source}, select={args.followon_select}.")

    if args.conversation_field and args.conversation_field.lower() == "none":
        conversation_field = None
    else:
        conversation_field = args.conversation_field or first_present(sample_row, CONVERSATION_KEYS)
    print(f"[data] Director conversation history from CRAFT column: {conversation_field or '(none)'}")

    policy = None
    if needs_policy:
        policy = PolicyScorer(
            args.policy_path, find_policy_code_dir(args.policy_path, args.policy_code_dir),
            feature_mode=args.policy_feature_mode, feature_tier=args.policy_feature_tier,
            allow_silent=args.policy_allow_silent,
        )
    if args.policy_check_only:
        args.horizon = 1   # nothing past the divergence turn is needed for the check

    director_model_name = args.director_model
    LocalDirectorAgent = None
    views_fn = None
    director_backend = None
    if args.followon_source == "generate" and args.horizon > 1:
        director_module = load_director_module(args.director_module)
        LocalDirectorAgent = make_local_director_class(director_module.DirectorAgent)
        views_fn = get_views_fn(director_module)
        director_backend = get_backend(args.director_backend)
        print(f"[directors] Generating follow-on utterances with {args.director_backend}/"
              f"'{director_model_name}' (DirectorAgent prompt from '{args.director_module}').")
    after_key = first_present(sample_row, BOARD_AFTER_KEYS)
    before_key = first_present(sample_row, BOARD_BEFORE_KEYS)
    if after_key:
        print(f"[data] Ground-truth board after each turn from column '{after_key}'.")
    elif before_key:
        print(f"[data] Ground-truth board after turn t from turn t+1's '{before_key}' "
              f"(last turn of each game will have no ground truth).")
    else:
        print("[warn] No board columns found in CRAFT; ground-truth trajectory metrics will be n/a.")

    run_cf = args.arms in ("both", "counterfactual")
    run_fa = args.arms in ("both", "factual")
    run_ra = not args.no_random_arm

    builder_backend = get_backend(args.backend)
    if not args.skip_preflight and not args.preview_director_prompt and not args.policy_check_only:
        preflight(builder_backend, args.model, args.base_url)
        if LocalDirectorAgent is not None and (args.director_backend, director_model_name) != \
                (args.backend, args.model):
            preflight(director_backend, director_model_name, args.base_url)

    with open(args.structures, "r", encoding="utf-8") as fh:
        struct_data = json.load(fh)
    target_by_id = {s["id"]: s["structure"] for s in struct_data}
    struct_by_id = {s["id"]: s for s in struct_data}

    oracle_regen = None
    if args.oracle_every_turn:
        oracle_regen = OracleRegenerator()
        missing = [s["id"] for s in struct_data
                   if target_spans_for(s) is None and structure_has_large_blocks(s["structure"])]
        if missing:
            sys.exit(f"[oracle] {len(missing)} structures have large blocks but no target spans "
                     f"(looked for {TARGET_SPAN_KEYS}). Keys in the first entry: "
                     f"{sorted(struct_data[0].keys())}. Without target spans, large-block "
                     f"placements can't be enumerated.")
        print("[oracle] --oracle-every-turn: step 0 uses CRAFT oracle_moves; follow-on turns "
              "regenerate candidates for each arm's board with oracle.py.")

    if LocalDirectorAgent is not None:
        # Fail now, not 20 turns in, if target views can't be built.
        try:
            compute_target_views(views_fn, struct_data[0])
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[directors] Could not build director target views ({exc}). "
                     f"Check get_director_views in structure_generator_v2.")

    def make_followon(row: Dict, struct_entry: Dict, select: str) -> Dict:
        target_views = None
        make_directors = None
        if LocalDirectorAgent is not None:
            target_views = compute_target_views(views_fn, struct_entry)

            def make_directors(r: Dict) -> Dict:
                return {
                    d: LocalDirectorAgent(
                        d, r.get(f"{d}_archetype"), director_backend, director_model_name,
                        temperature=args.director_temperature,
                        max_tokens=args.director_max_tokens,
                        seed=None if args.seed < 0 else args.seed,
                        fallback_key=str(r.get("structure_id")),
                    )
                    for d in DIRECTORS
                }
        return {
            "source": args.followon_source,
            "select": select,
            "policy": policy,
            "step0_mode": step0_mode,
            "recommend_template": args.recommend_template,
            "random_seed": args.random_seed,
            "oracle": oracle_regen,
            "target_raw": struct_entry["structure"],
            "target_spans": target_spans_for(struct_entry),
            "sat_fn": make_sat_fn(struct_entry["structure"]),
            "conversation_field": conversation_field,
            "max_history_chars": args.max_history_chars or None,
            "target_views": target_views,
            "make_directors": make_directors,
        }

    turns = load_divergence_turns(args.divergence_dir)
    if not args.no_shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(turns)
        print(f"[data] Shuffled {len(turns)} turns with shuffle-seed={args.shuffle_seed}.")

    if args.decidability_file:
        decidable_keys = load_decidable_keys(args.decidability_file)
        before = len(turns)

        def is_decidable_turn(turn):
            cf_msg = turn.get(f"{turn['predicted']}_message")
            if cf_msg is None:
                return False
            key = (cf_msg["timestamp"], cf_msg["structure"],
                   normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip()))
            return key in decidable_keys

        turns = [t for t in turns if is_decidable_turn(t)]
        print(f"[data] --decidability-file: kept {len(turns)}/{before} turns marked decidable.")

    if args.limit:
        turns = turns[: args.limit]

    if writes_output and "anthropic" in (args.backend, args.director_backend):
        arm_selects = {}
        if run_cf:
            arm_selects["counterfactual"] = cf_select
        if run_fa:
            arm_selects["factual"] = fa_select
        if run_ra:
            arm_selects["random"] = "random"
        b_calls, d_calls = estimate_max_calls(
            len(turns), len(arm_selects), args.horizon,
            args.followon_source if LocalDirectorAgent is not None else "history",
            arm_selects, args.max_retries)
        print(f"[cost] Upper bound for {len(turns)} turns: ~{b_calls} Builder calls (incl. format "
              f"retries) and ~{d_calls} director calls. Non-divergence turns, early stops and "
              f"parseable first answers make the real number lower.")

    agent = LocalBuilderAgent(
        backend=builder_backend,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=None if args.seed < 0 else args.seed,
        max_retries=args.max_retries,
    )

    saved_by_game: Dict[str, List[Dict]] = {}
    done: Dict[tuple, Dict] = {}
    if writes_output and args.resume:
        if args.out_dir.exists() and game_files(args.out_dir):
            prev = saved_run_args(args.out_dir)
            if prev is None:
                print("[resume] WARNING: no saved settings found in this folder; cannot confirm the "
                      "saved results used the same settings.")
            else:
                conflicts = resume_conflicts(prev, serializable_args(args))
                if conflicts:
                    sys.exit("[resume] Not resuming: these settings differ from the saved run:\n" +
                             "\n".join(f"    {k}: saved={a!r}  now={b!r}" for k, (a, b) in conflicts.items()) +
                             "\n  Use the original settings, or start a new run without --resume.")
            saved_by_game = load_saved_records(args.out_dir)
            for rows in saved_by_game.values():
                for r in rows:
                    done[record_identity(r)] = r
            print(f"[resume] {len(done)} completed turns found in {len(saved_by_game)} game files "
                  f"in {args.out_dir}; they will be skipped.")
        else:
            print(f"[resume] Nothing saved yet in {args.out_dir}; starting from the beginning.")

    if writes_output:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.out_dir / RUN_CONFIG_NAME, {"run_args": serializable_args(args)})

    # Pre-loaded with saved results, so rewriting a game's file never drops them.
    all_turns_counter: Dict[str, List[Dict]] = defaultdict(
        list, {k: list(v) for k, v in saved_by_game.items()})
    stats: Dict[str, int] = defaultdict(int)
    # pair -> step -> endpoint -> values, for every (minuend, subtrahend) in ARM_PAIRS
    deltas_by_pair = {pr: defaultdict(lambda: defaultdict(list)) for pr in ARM_PAIRS}
    # pair -> endpoint -> values, turns where both arms of the pair reached the full horizon
    full_horizon_by_pair = {pr: defaultdict(list) for pr in ARM_PAIRS}
    # counterfactual − factual keeps its original names for downstream analysis
    deltas_by_step = deltas_by_pair[("counterfactual", "factual")]
    # step -> metric -> values (absolute, per arm, vs ground truth)
    gt_by_step: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    # turns where both arms reached the full horizon: step0 vs final on the SAME turns
    full_horizon = full_horizon_by_pair[("counterfactual", "factual")]
    deltas_by_director: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    deltas_by_move_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    horizon_reached_counts: Dict[int, int] = defaultdict(int)
    oracle_adherence: Dict[int, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
    t_start = time.time()

    def aggregate(record: Dict, verbose: bool = True) -> None:
        """Fold one finished divergence turn into the run statistics. Used for
        new turns and for turns loaded from disk by --resume, so the summary is
        identical either way."""
        predicted = record["counterfactualDirector"]
        cf_silent = record["silent_director"]
        director_label = record["director_model_label"]
        if record.get("random"):
            rd = record.get("randomDirector")
            stats["random_arm_n"] += 1
            stats["random_step0_equals_history"] += int(rd == record["historicalDirector"])
            stats["random_step0_equals_policy"] += int(rd == predicted)
        for arm_name, prefix in ARM_PREFIX.items():
            for s in (record.get(arm_name) or {}).get("steps", []):
                stats[f"{prefix}_action_{s['builder_move']['action']}"] += 1
                stats[f"{prefix}_parse_failure"] += int(s["parse_failed"])
                mode = s.get("selection", {}).get("mode")
                if mode in POLICY_SELECTS:
                    stats[f"{prefix}_policy_steps"] += 1
                    stats[f"{prefix}_policy_picked_silent"] += int(s["selection"]["picked_silent"])
                elif mode == "random":
                    stats[f"{prefix}_random_steps"] += 1
                if s.get("oracle_adherent") is not None:
                    oracle_adherence[s["step"]][arm_name].append(bool(s["oracle_adherent"]))
                if s.get("span_recorded") is False:
                    stats["span_not_recorded"] += 1
                if s.get("progress_vs_ground_truth") is not None:
                    gt_by_step[s["step"]][f"{arm_name}_progress_vs_ground_truth"].append(
                        s["progress_vs_ground_truth"])
                    gt_by_step[s["step"]][f"{arm_name}_iou_vs_ground_truth_board"].append(
                        s["iou_vs_ground_truth_board"])

        record["delta_pairs"] = {}
        for a, b in ARM_PAIRS:
            if not record.get(a) or not record.get(b):
                continue
            a_steps, b_steps = record[a]["steps"], record[b]["steps"]
            n_common = min(len(a_steps), len(b_steps))
            if n_common == 0:
                continue
            store = deltas_by_pair[(a, b)]
            per_step = []
            for k in range(n_common):
                c, f = a_steps[k], b_steps[k]
                d_full = c["satisfaction_full"] - f["satisfaction_full"]
                # Always the POLICY director's view, so all pairs use one yardstick.
                d_policy = c["partial_views"][predicted] - f["partial_views"][predicted]
                entry = {"step": k, "turn": c["turn"], "full": d_full,
                         "partial_policy_view": d_policy, "iou_vs_ground_truth_board": None}
                store[k]["full_board"].append(d_full)
                store[k]["partial_policy_view"].append(d_policy)
                if not cf_silent:
                    store[k]["full_board_excluding_silent"].append(d_full)
                if c["iou_vs_ground_truth_board"] is not None and f["iou_vs_ground_truth_board"] is not None:
                    d_gt = c["iou_vs_ground_truth_board"] - f["iou_vs_ground_truth_board"]
                    entry["iou_vs_ground_truth_board"] = d_gt
                    store[k]["iou_vs_ground_truth_board"].append(d_gt)
                per_step.append(entry)
                if verbose:
                    print(f"    {ARM_PREFIX[a]}-{ARM_PREFIX[b]} t+{k}: delta_full={d_full:+.4f}  "
                          f"delta_policy_view={d_policy:+.4f}  "
                          f"delta_iou_vs_gt={_fmt(entry['iou_vs_ground_truth_board'])}")

            reached_full = n_common == args.horizon
            record["delta_pairs"][f"{a}_minus_{b}"] = {
                "steps_compared": n_common, "full_horizon": reached_full, "per_step": per_step}
            step0, final = per_step[0], per_step[-1]
            if reached_full:
                fh = full_horizon_by_pair[(a, b)]
                fh["step0_full_board"].append(step0["full"])
                fh["final_full_board"].append(final["full"])
                fh["step0_partial_policy_view"].append(step0["partial_policy_view"])
                fh["final_partial_policy_view"].append(final["partial_policy_view"])
                if final["iou_vs_ground_truth_board"] is not None:
                    fh["final_iou_vs_ground_truth_board"].append(final["iou_vs_ground_truth_board"])
                for arm_name, steps in ((a, a_steps), (b, b_steps)):
                    p = steps[-1]["progress_vs_ground_truth"]
                    if p is not None:
                        fh[f"final_{arm_name}_progress_vs_ground_truth"].append(p)

            if (a, b) != ("counterfactual", "factual"):
                continue
            # counterfactual − factual only: the original per-turn fields and breakdowns.
            horizon_reached_counts[n_common] += 1
            move_type_key = f"{a_steps[0]['builder_move']['action']}/{b_steps[0]['builder_move']['action']}"
            record["move_type"] = move_type_key
            record["delta"] = {
                "steps_compared": n_common,
                "full_horizon": reached_full,
                "progress_before": a_steps[0]["satisfaction_full_before"],
                "per_step": per_step,
            }
            for bucket in (deltas_by_director[director_label], deltas_by_move_type[move_type_key]):
                bucket["step0_full_board"].append(step0["full"])
                bucket["step0_partial_policy_view"].append(step0["partial_policy_view"])
                if reached_full:
                    bucket["final_full_board"].append(final["full"])
                    bucket["final_partial_policy_view"].append(final["partial_policy_view"])
                    if final["iou_vs_ground_truth_board"] is not None:
                        bucket["final_iou_vs_ground_truth_board"].append(final["iou_vs_ground_truth_board"])

    for count, turn in enumerate(turns, start=1):
        predicted = turn["predicted"]
        historical = turn.get("builderSelected")
        print(f"\nTurn {count}/{len(turns)}  policy={predicted}  history={historical}")

        if historical is None:
            print("  [skip] no builderSelected field — cannot pair a factual arm")
            stats["missing_history"] += 1
            continue
        if predicted == historical:
            print("  [skip] policy agreed with history — not a divergence turn")
            stats["non_divergence"] += 1
            continue

        cf_director = predicted
        cf_msg = turn.get(f"{cf_director}_message")
        fa_msg = turn.get(f"{historical}_message")
        if cf_msg is None or fa_msg is None:
            print("  [skip] missing message object for one arm")
            stats["missing_message"] += 1
            continue

        cf_silent = is_silent(cf_msg)
        if cf_silent:
            stats["silent_director"] += 1
            print(f"  [silent] {cf_director} produced no utterance this turn")
            if args.skip_silent:
                continue
        if is_silent(fa_msg):
            print(f"  [skip] historical director {historical} is silent — log inconsistency")
            stats["silent_history"] += 1
            continue

        turn_index = cf_msg["timestamp"]
        structure_id = cf_msg["structure"]
        director_model = normalize_director_model(cf_msg["modelCombo"].split("+")[0].strip())

        if fa_msg.get("structureBefore") != cf_msg.get("structureBefore"):
            print("  [skip] arms disagree on structureBefore — cannot pair")
            stats["board_mismatch"] += 1
            continue

        # Random control arm's step-0 director (--random-step0).
        ra_director, ra_msg = None, None
        if run_ra:
            if args.random_step0 == "policy":
                ra_director = predicted
            else:
                pool = [d for d in DIRECTORS if not is_silent(turn.get(f"{d}_message"))]
                if args.random_step0 == "nonhistory":
                    pool = [d for d in pool if d != historical]
                if pool:
                    ra_director = random_draw(args.random_seed, "step0", structure_id, turn_index,
                                              cf_msg["modelCombo"], pool=pool)
            if ra_director is not None:
                ra_msg = turn.get(f"{ra_director}_message")
                if ra_msg is not None and ra_msg.get("structureBefore") != cf_msg.get("structureBefore"):
                    ra_msg = None
            if ra_msg is None:
                stats["random_arm_skipped"] += 1
                print(f"  [random] no eligible director for the random arm's step 0 "
                      f"(--random-step0 {args.random_step0}); running the other arms only")
            else:
                print(f"  [random] step-0 director {ra_director}"
                      + (" (same as history)" if ra_director == historical else "")
                      + (" (same as policy)" if ra_director == predicted else ""))

        row = craft_index.get((turn_index, structure_id, director_model))
        if row is None:
            print(f"  [skip] no CRAFT row for {turn_index}, {structure_id}, {director_model}")
            stats["no_craft_row"] += 1
            continue

        target = target_by_id.get(row["structure_id"])
        if target is None:
            print(f"  [skip] no target structure for id {row['structure_id']}")
            stats["no_target"] += 1
            continue
        target_norm = normalize_structure(target)
        oracle_moves = json.loads(row["oracle_moves"]) if args.use_oracle else None
        director_label = director_label_from_combo(cf_msg["modelCombo"])

        oracle_check = None
        if oracle_regen is not None:
            # Regenerate on the historical pre-move board and compare with what
            # CRAFT logged, to validate the regeneration used at follow-on turns.
            regen, _ = oracle_regen.candidates(
                parse_board(cf_msg["structureBefore"]), json.loads(row["spans_before"]),
                target, target_spans_for(struct_by_id[row["structure_id"]]))
            logged_keys = {oracle_move_key(c) for c in usable_candidates(oracle_moves)}
            regen_keys = {oracle_move_key(c) for c in regen}
            oracle_check = {
                "logged": len(logged_keys), "regenerated": len(regen_keys),
                "match": logged_keys == regen_keys,
                "only_logged": [list(k) for k in sorted(logged_keys - regen_keys, key=str)],
                "only_regenerated": [list(k) for k in sorted(regen_keys - logged_keys, key=str)],
            }
            stats["oracle_check_n"] += 1
            stats["oracle_check_agree"] += int(oracle_check["match"])
            print(f"  [oracle check] logged={len(logged_keys)} regenerated={len(regen_keys)} "
                  f"{'MATCH' if oracle_check['match'] else 'DIFFER'}")

        policy_check = None
        if policy is not None:
            # Rebuild exactly what the policy saw when it made the logged pick:
            # every director's history up to and including turn t, with the
            # historical after-board, then compare argmax to `predicted`.
            sat_fn = make_sat_fn(target)
            after_fn = lambda s_: ground_truth_board_after(  # noqa: E731
                craft_index, s_, structure_id, director_model, after_key)
            hist = policy.craft_histories(craft_index, structure_id, director_model,
                                          turn_index + 1, cf_msg.get("modelCombo"), sat_fn, after_fn)
            scores = policy.score(hist)
            raw = max(DIRECTORS, key=lambda d: scores[d])
            text_match = all(
                (turn.get(f"{d}_message") or {}).get("utterance", row.get(f"{d}_message") or "")
                == (row.get(f"{d}_message") or "")
                for d in DIRECTORS
            )
            policy_check = {"scores": scores, "argmax": raw, "logged_predicted": predicted,
                            "reproduced": raw == predicted, "utterances_match_craft": text_match}
            stats["policy_check_n"] += 1
            stats["policy_check_agree"] += int(raw == predicted)
            stats["policy_check_text_mismatch"] += int(not text_match)
            print(f"  [policy check] argmax={raw} logged={predicted} "
                  f"{'OK' if raw == predicted else 'MISMATCH'}"
                  + ("" if text_match else "  (divergence-log utterances differ from CRAFT text)"))
            if args.policy_check_only:
                continue

        ident = (turn_index, str(row["structure_id"]), cf_msg["modelCombo"], cf_director, historical)
        if ident in done:
            aggregate(done.pop(ident), verbose=False)
            stats["resumed"] += 1
            stats["scored"] += 1
            print("  [resume] already saved; using the saved result")
            continue

        record: Dict = {
            "turn": turn_index,
            "structure_id": row["structure_id"],
            "modelCombo": cf_msg["modelCombo"],
            "director_model_label": director_label,
            "counterfactualDirector": cf_director,
            "historicalDirector": historical,
            "randomDirector": ra_director if ra_msg is not None else None,
            "silent_director": cf_silent,
            "horizon": args.horizon,
            "followon_source": args.followon_source,
            "followon_select": cf_select,
            "factual_followon_select": fa_select,
            "step0_mode": step0_mode,
            "recommend_template": args.recommend_template if "recommend" in (cf_select, fa_select) else None,
            "random_arm": run_ra,
            "random_step0": args.random_step0 if run_ra else None,
            "random_seed": args.random_seed if run_ra else None,
            "policy_check": policy_check,
            "oracle_every_turn": args.oracle_every_turn,
            "oracle_check": oracle_check,
            "director_generation_model": director_model_name if LocalDirectorAgent else None,
            "director_backend": args.director_backend if LocalDirectorAgent else None,
            "builder_backend": args.backend,
            "builder_model": args.model,
            "seed": args.seed if builder_backend.supports_seed else None,
            "shuffle_seed": None if args.no_shuffle else args.shuffle_seed,
        }

        try:
            followon_cf = make_followon(row, struct_by_id[row["structure_id"]], cf_select)
            followon_cf["step0_source"] = {
                "counterfactual": "policy", "factual": "history",
                "random": "policy" if args.random_step0 == "policy" else f"random:{args.random_step0}"}
            followon_fa = {**followon_cf, "select": fa_select}
            followon = followon_cf
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] could not build director target views: {exc}")
            stats["no_target_views"] += 1
            continue

        if args.preview_director_prompt:
            if LocalDirectorAgent is None:
                sys.exit("--preview-director-prompt needs --followon-source generate and --horizon > 1.")
            d1 = followon["make_directors"](row)["D1"]
            board0 = parse_board(cf_msg["structureBefore"])
            convo = base_conversation(row, conversation_field)
            convo = "\n".join([c for c in [convo, "Builder: <this arm's turn-t move goes here>"] if c])
            print("\n" + "=" * 76)
            print(f"  D1 PROMPT PREVIEW  (turn {turn_index + 1}, archetype={d1.archetype}, "
                  f"source={d1.archetype_source})")
            print("=" * 76)
            print(d1.build_prompt(board0, followon["target_views"]["D1"],
                                  tail(convo, args.max_history_chars or None)))
            return

        PROMPT_LOG["print"] = stats["scored"] < args.print_prompts

        rollout_kwargs = dict(
            row=row, target_norm=target_norm, oracle_moves=oracle_moves,
            craft_index=craft_index, turn_index=turn_index, structure_id=structure_id,
            director_model=director_model, horizon=args.horizon,
            after_key=after_key,
        )

        if run_cf:
            record["counterfactual"] = run_rollout(
                agent, "counterfactual", cf_director, cf_msg,
                followon=followon_cf, check_tokens=(count == 1), **rollout_kwargs,
            )

        if run_fa:
            # Same Builder, same seed (Ollama), same prompts, same follow-on
            # dialogue; only the step-0 director differs.
            record["factual"] = run_rollout(
                agent, "factual", historical, fa_msg, followon=followon_fa, **rollout_kwargs,
            )

        if ra_msg is not None:
            # Random control: same format as the counterfactual arm, but every
            # director choice is a uniform random draw.
            followon_ra = {**followon_cf, "select": "random",
                           "random_as_recommend": cf_select == "recommend"}
            record["random"] = run_rollout(
                agent, "random", ra_director, ra_msg, followon=followon_ra, **rollout_kwargs,
            )

        aggregate(record)

        # Save this game's file now, so an interruption keeps every finished turn.
        key = f"{cf_msg['modelCombo']}_{row['structure_id']}"
        all_turns_counter[key].append(record)
        write_json_atomic(args.out_dir / f"{key}.json", all_turns_counter[key])
        stats["scored"] += 1

    if args.policy_check_only:
        n, a = stats["policy_check_n"], stats["policy_check_agree"]
        print("\n" + "=" * 76)
        print(f"  POLICY REPRODUCTION CHECK: {a}/{n} divergence turns reproduced "
              f"({(a / n if n else float('nan')):.1%})")
        if stats["policy_check_text_mismatch"]:
            print(f"  {stats['policy_check_text_mismatch']} turns had divergence-log utterances that "
                  f"differ from the CRAFT text -- likely the cause of mismatches there.")
        print("  Near 100%: features are reproduced; the policy can be trusted in rollouts.\n"
              "  Well below: config (FEATURE_MODE/TIER, USE_POOLED_HISTORY) or turn numbering\n"
              "  differs from how this policy was trained. Fix before running rollouts.")
        print("=" * 76)
        return

    # ── Write ──────────────────────────────────────────────────────────────
    # Files are already written after every turn; this final pass is a safety net.
    for key, rows in all_turns_counter.items():
        write_json_atomic(args.out_dir / f"{key}.json", rows)

    processed = stats["scored"]
    elapsed = time.time() - t_start
    silent_rate = stats["silent_director"] / max(len(turns), 1)

    summary = {
        "out_dir": str(args.out_dir),
        "run_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())
                     if k != "anthropic_api_key"},
        "builder_backend": args.backend,
        "builder_model": args.model,
        "director_backend": args.director_backend if LocalDirectorAgent else None,
        "base_url": args.base_url if "ollama" in backends else None,
        "seed": args.seed if builder_backend.supports_seed else None,
        "api_usage": {name: b.usage_summary() for name, b in backends.items()},
        "arms": args.arms,
        "horizon": args.horizon,
        "followon_source": args.followon_source,
        "followon_select": cf_select,
        "factual_followon_select": fa_select,
        "step0_mode": step0_mode,
        "recommend_template": args.recommend_template if "recommend" in (cf_select, fa_select) else None,
        "random_arm": ({
            "random_seed": args.random_seed,
            "random_step0": args.random_step0,
            "mirrors_counterfactual_rule": cf_select,
            "turns_with_random_arm": stats["random_arm_n"],
            "turns_without_random_arm": stats["random_arm_skipped"],
            "followon_random_steps": stats["ra_random_steps"],
            "step0_draw_equals_history": stats["random_step0_equals_history"],
            "step0_draw_equals_policy": stats["random_step0_equals_policy"],
        } if run_ra else None),
        "save_prompts": args.save_prompts,
        "policy_path": str(args.policy_path) if args.policy_path else None,
        "policy_feature_mode": policy.cfg.FEATURE_MODE if policy else None,
        "policy_feature_tier": policy.cfg.FEATURE_TIER if policy else None,
        "policy_followon_steps": {
            "counterfactual": stats["cf_policy_steps"], "factual": stats["fa_policy_steps"],
            "argmax_was_silent_counterfactual": stats["cf_policy_picked_silent"],
            "argmax_was_silent_factual": stats["fa_policy_picked_silent"],
        } if policy else None,
        "policy_reproduction": ({
            "n": stats["policy_check_n"], "agree": stats["policy_check_agree"],
            "rate": round(stats["policy_check_agree"] / stats["policy_check_n"], 4)
            if stats["policy_check_n"] else None,
        } if policy else None),
        "director_generation_model": director_model_name if LocalDirectorAgent else None,
        "director_module": args.director_module if LocalDirectorAgent else None,
        "conversation_field": conversation_field,
        "ground_truth_board_source": after_key or (f"next turn's {before_key}" if before_key else None),
        "decidability_file": str(args.decidability_file) if args.decidability_file else None,
        "use_oracle": args.use_oracle,
        "oracle_every_turn": args.oracle_every_turn,
        "oracle_regeneration_check": ({
            "n": stats["oracle_check_n"], "match": stats["oracle_check_agree"],
            "rate": round(stats["oracle_check_agree"] / stats["oracle_check_n"], 4)
            if stats["oracle_check_n"] else None,
        } if args.oracle_every_turn else None),
        "oracle_adherence_by_step": {
            f"t+{k}": {arm: {"n": len(v), "rate": round(sum(v) / len(v), 4) if v else None}
                       for arm, v in sorted(arms.items())}
            for k, arms in sorted(oracle_adherence.items())
        } if args.use_oracle else None,
        "large_spans_not_recorded": stats["span_not_recorded"] if args.oracle_every_turn else None,
        "divergence_dir": str(args.divergence_dir),
        "turns_seen": len(turns),
        "turns_scored": processed,
        "turns_resumed_from_disk": stats["resumed"],
        "saved_turns_not_in_this_run": len(done),
        "games_written": len(all_turns_counter),
        "steps_compared_histogram": dict(sorted(horizon_reached_counts.items())),
        "silent_director_count": stats["silent_director"],
        "silent_director_rate": round(silent_rate, 4),
        "skip_silent": args.skip_silent,
        "parse_failures": {"counterfactual": stats["cf_parse_failure"], "factual": stats["fa_parse_failure"],
                           "random": stats["ra_parse_failure"] if run_ra else None},
        "action_counts": {k: v for k, v in stats.items() if "_action_" in k},
        "skipped": {
            k: stats[k] for k in (
                "missing_history", "non_divergence", "missing_message",
                "silent_history", "board_mismatch", "no_craft_row", "no_target",
                "no_target_views",
            ) if stats[k]
        },
        "elapsed_sec": round(elapsed, 1),
        "sec_per_turn": round(elapsed / max(processed, 1), 2),
    }

    def by_step(store):
        return {f"t+{k}": {name: pack(vals) for name, vals in endpoints.items()}
                for k, endpoints in sorted(store.items())}

    if gt_by_step:
        summary["ground_truth_by_step"] = {
            f"t+{k}": {name: pack(vals) for name, vals in metrics.items()}
            for k, metrics in sorted(gt_by_step.items())
        }
    for a, b in ARM_PAIRS[1:]:
        if deltas_by_pair[(a, b)]:
            summary[f"paired_delta_by_step_{a}_minus_{b}"] = by_step(deltas_by_pair[(a, b)])
            summary[f"full_horizon_turns_{a}_minus_{b}"] = {
                name: pack(vals) for name, vals in full_horizon_by_pair[(a, b)].items()}

    if run_cf and run_fa:
        summary["paired_delta_by_step"] = by_step(deltas_by_step)
        summary["full_horizon_turns"] = {name: pack(vals) for name, vals in full_horizon.items()}
        summary["paired_delta_by_director"] = {
            label: {name: pack(vals) for name, vals in endpoints.items()}
            for label, endpoints in sorted(deltas_by_director.items())
        }
        summary["paired_delta_by_move_type"] = {
            label: {name: pack(vals) for name, vals in endpoints.items()}
            for label, endpoints in sorted(deltas_by_move_type.items())
        }

    with open(args.out_dir / f"_run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 76)
    print(f"  BUILDER ROLLOUT SUMMARY  ({args.backend}/{args.model}, horizon={args.horizon}, "
          f"follow-on={args.followon_source}, cf={cf_select}, fa={fa_select}; counterfactual − factual)")
    print("=" * 76)
    for k, v in summary.items():
        if k == "run_args":
            continue
        if not k.startswith(("paired_delta", "ground_truth_by_step", "full_horizon_turns",
                             "random_arm")):
            print(f"  {k:<28} {v}")
    if summary.get("random_arm"):
        print(f"  {'random_arm':<28} {summary['random_arm']}")
    if "paired_delta_by_step" in summary:
        print("  " + "-" * 72)
        print("  PAIRED DELTA BY STEP  (counterfactual − factual)")
        for step, endpoints in summary["paired_delta_by_step"].items():
            print_table(step, endpoints, indent="    ")
    for a, b, meaning in (("random", "factual", "random selection vs history"),
                          ("counterfactual", "random", "policy vs random selection, same information")):
        key = f"paired_delta_by_step_{a}_minus_{b}"
        if key in summary:
            print("  " + "-" * 72)
            print(f"  PAIRED DELTA BY STEP  ({a} − {b}: {meaning})")
            for step, endpoints in summary[key].items():
                print_table(step, endpoints, indent="    ")
    if "ground_truth_by_step" in summary:
        print("  " + "-" * 72)
        print("  EACH ARM vs GROUND-TRUTH TRAJECTORY BY STEP")
        print("  (progress_vs_ground_truth > 0 means the arm is closer to the target than history was)")
        for step, metrics in summary["ground_truth_by_step"].items():
            print_table(step, metrics, indent="    ")
    if "full_horizon_turns" in summary:
        print("  " + "-" * 72)
        print_table("TURNS REACHING FULL HORIZON (step 0 vs final, same turns; counterfactual − factual)",
                    summary["full_horizon_turns"])
    for a, b in ARM_PAIRS[1:]:
        key = f"full_horizon_turns_{a}_minus_{b}"
        if summary.get(key):
            print("  " + "-" * 72)
            print_table(f"TURNS REACHING FULL HORIZON ({a} − {b})", summary[key])
    if "paired_delta_by_director" in summary:
        print("  " + "-" * 72)
        print("  BY DIRECTOR MODEL")
        for label, endpoints in summary["paired_delta_by_director"].items():
            print_table(label, endpoints, indent="    ")
        print("  " + "-" * 72)
        print("  BY STEP-0 MOVE TYPE (counterfactual/factual)")
        for label, endpoints in summary["paired_delta_by_move_type"].items():
            print_table(label, endpoints, indent="    ")
    print("=" * 76)
    if silent_rate > 0.10:
        print(
            f"  ⚠ Silent-director rate {silent_rate:.1%} exceeds 10%. Compare 'full_board'\n"
            f"    against 'full_board_excluding_silent' before reading delta as policy quality."
        )
    if run_ra and stats["random_arm_n"]:
        print(f"  ℹ Random arm's step-0 director matched history's on "
              f"{stats['random_step0_equals_history']}/{stats['random_arm_n']} turns and the "
              f"policy's on {stats['random_step0_equals_policy']}/{stats['random_arm_n']}.")
    if policy is not None and stats["policy_check_n"] and \
            stats["policy_check_agree"] / stats["policy_check_n"] < 0.9:
        print(f"  ⚠ The loaded policy reproduced only {stats['policy_check_agree']}/"
              f"{stats['policy_check_n']} logged divergence picks. Its follow-on choices may not be\n"
              f"    the trained policy's. Run --policy-check-only and fix the config first.")
    if args.oracle_every_turn and stats["oracle_check_n"] and \
            stats["oracle_check_agree"] / stats["oracle_check_n"] < 0.9:
        print(f"  ⚠ Regenerated oracle moves matched CRAFT's logged ones on only "
              f"{stats['oracle_check_agree']}/{stats['oracle_check_n']} divergence turns. Check\n"
              f"    'oracle_check' in the records (target spans / spans format) before trusting\n"
              f"    the follow-on oracle candidates.")
    for name, b in backends.items():
        for role, u in b.usage_summary().items():
            if role == "params_dropped" or not isinstance(u, dict):
                continue
            if u.get("max_tokens_stops"):
                print(f"  ⚠ {u['max_tokens_stops']} {name} {role} responses hit max_tokens; check "
                      f"parse_failures and raise the token limit if they are high.")
    print(f"  Wrote {len(all_turns_counter)} game files to {args.out_dir}")


if __name__ == "__main__":
    main()