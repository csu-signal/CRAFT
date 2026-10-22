#!/usr/bin/env python3
"""
prepare_hf_model.py
===================

Turn a Hub repo that `transformers` refuses to load ("Unrecognized model ...
Should have a `model_type` key in its config.json") into a local checkpoint
directory that `run_single_builder_local.py --backend hf-local` can use.

Handles:
  * Checkpoint in a SUBFOLDER (e.g. checkpoint-350/) -> found automatically,
    or chosen with --subfolder when there are several.
  * LoRA / PEFT adapter (adapter_config.json) -> merged into its base model
    and saved as full weights.
  * config.json missing model_type / architecture fields -> missing keys
    filled in from the base model's config, weights copied unchanged.
  * GGUF-only repo -> prints the Ollama command instead.
  * Sharded FSDP training checkpoint (e.g. verl's *rank_*.pt) -> explains it
    must be merged to HF format with the training framework's merger first.

A missing tokenizer or chat template is taken from the base model, and the
result is verified with AutoConfig/AutoTokenizer at the end.

Usage
-----
    pip install transformers accelerate huggingface_hub peft
    python prepare_hf_model.py <repo> --inspect-only          # list files, download nothing
    python prepare_hf_model.py <repo> --out ./fixed            # auto-detect subfolder
    python prepare_hf_model.py <repo> --out ./fixed --subfolder checkpoint-350
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
MARKERS = ("config.json", "adapter_config.json")


def checkpoint_dirs(files):
    """Directories (relative, '' = repo root) containing a config or adapter config."""
    dirs = set()
    for f in files:
        p = PurePosixPath(f)
        if p.name in MARKERS and not any(part.startswith(".") for part in p.parts):
            dirs.add(str(p.parent) if str(p.parent) != "." else "")
    return sorted(dirs)


def in_dir(files, d):
    return [f for f in files if (f.startswith(d + "/") if d else "/" not in f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="Hub repo id (or a local directory)")
    ap.add_argument("--out", type=Path, default=None, help="Output directory for the fixed checkpoint")
    ap.add_argument("--subfolder", default=None,
                    help="Subfolder holding the checkpoint (auto-detected when there is only one)")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="Base model used to fill missing config/tokenizer and, for adapters "
                         "with a quantized or missing base, to merge into.")
    ap.add_argument("--token", default=None, help="HF token (private/gated repos); defaults to cached login")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    local = Path(args.repo).is_dir()

    # ---- List first (no download) -------------------------------------------
    if local:
        root = Path(args.repo)
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    else:
        files = sorted(list_repo_files(args.repo, token=args.token))
    print(f"[inspect] {args.repo} contains {len(files)} file(s):")
    for f in files:
        print(f"    {f}")

    candidates = checkpoint_dirs(files)
    print(f"[inspect] checkpoint folders found: {[c or '(root)' for c in candidates] or 'none'}")

    # ---- Choose the folder ----------------------------------------------------
    if args.subfolder is not None:
        sub = args.subfolder.strip("/")
    elif len(candidates) == 1:
        sub = candidates[0]
    elif len(candidates) > 1:
        sys.exit("[result] several checkpoint folders; re-run with --subfolder <one of the above> "
                 "(usually the highest step / the one with weight files).")
    else:
        sub = None

    if sub is None:
        gguf = [f for f in files if f.endswith(".gguf")]
        fsdp = [f for f in files if f.endswith(".pt") and "rank" in PurePosixPath(f).name]
        if gguf:
            sys.exit(
                "[result] GGUF repo -- transformers can't load it. Use the Ollama backend:\n"
                f"    ollama pull hf.co/{args.repo}\n"
                f"    python run_single_builder_local.py --backend ollama --model hf.co/{args.repo} ..."
            )
        if fsdp:
            sys.exit(
                "[result] This looks like a sharded FSDP training checkpoint (e.g. verl), not a\n"
                "         Hugging Face model. Merge it to HF format with the training framework's\n"
                "         merger first (for verl: `python -m verl.model_merger merge --backend fsdp\n"
                "         --local_dir <.../actor> --target_dir <out>`; older versions use\n"
                "         scripts/model_merger.py), then pass that output dir to --model."
            )
        sys.exit("[result] no config.json or adapter_config.json anywhere in the repo, and no\n"
                 "         recognizable weight format. Paste the file list above to diagnose.")

    sub_files = in_dir(files, sub)
    label = sub or "(root)"
    print(f"[inspect] using folder {label} ({len(sub_files)} files)")
    for name in MARKERS:
        rel = f"{sub}/{name}" if sub else name
        if rel in files:
            path = (Path(args.repo) / rel) if local else Path(
                hf_hub_download(args.repo, rel, token=args.token))
            print(f"[inspect] {rel}:\n{path.read_text()}")

    if args.inspect_only:
        return
    if args.out is None:
        sys.exit("[error] --out is required unless --inspect-only")

    # ---- Download just that folder -------------------------------------------
    if local:
        src = Path(args.repo) / sub if sub else Path(args.repo)
    else:
        patterns = [f"{sub}/*"] if sub else [f for f in sub_files]
        src_root = Path(snapshot_download(args.repo, token=args.token, allow_patterns=patterns))
        src = src_root / sub if sub else src_root

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg_path, adapter_path = src / "config.json", src / "adapter_config.json"
    args.out.mkdir(parents=True, exist_ok=True)

    if adapter_path.exists():
        # ---- LoRA adapter -> merge into base ---------------------------------
        try:
            from peft import PeftModel
        except ImportError:
            sys.exit("[error] adapter checkpoint: pip install peft")
        import torch

        base = json.loads(adapter_path.read_text()).get("base_model_name_or_path") or args.base
        if "bnb" in base.lower() or "4bit" in base.lower():
            print(f"[adapter] base '{base}' is quantized; merging into '{args.base}' instead.")
            base = args.base
        print(f"[adapter] loading base '{base}' in bf16 on CPU (needs ~16 GB RAM) ...")
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="cpu", token=args.token,
        )
        tok_src = src if (src / "tokenizer_config.json").exists() else base
        tok = AutoTokenizer.from_pretrained(tok_src, token=args.token)
        # Only GROW: Qwen pads its embedding matrix (152064) beyond the
        # tokenizer size (~151665), so a mismatch alone is normal and
        # shrinking would change the model. Grow only if the fine-tune added
        # tokens beyond the padded size.
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok))
        print("[adapter] applying and merging adapter ...")
        model = PeftModel.from_pretrained(model, src).merge_and_unload()
        model.save_pretrained(args.out, safe_serialization=True)
        tok.save_pretrained(args.out)

    else:
        # ---- Full checkpoint: copy, fill any missing config keys -------------
        shutil.copytree(src, args.out, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".cache", ".git*"))
        src_cfg = json.loads(cfg_path.read_text())
        if not src_cfg.get("model_type"):
            base_cfg = AutoConfig.from_pretrained(args.base, token=args.token).to_dict()
            filled = sorted(k for k in base_cfg if k not in src_cfg)
            merged = {**base_cfg, **src_cfg, "model_type": base_cfg["model_type"]}
            (args.out / "config.json").write_text(json.dumps(merged, indent=2))
            print(f"[config] filled {len(filled)} missing key(s) from {args.base}: {filled}")
            print("         Check hidden_size / num_hidden_layers / vocab_size match the weights.")
        else:
            print(f"[config] config.json is fine (model_type='{src_cfg['model_type']}'); "
                  f"the original error was only the subfolder.")

    # ---- Tokenizer / chat template --------------------------------------------
    if not (args.out / "tokenizer_config.json").exists():
        print(f"[tokenizer] none in checkpoint; copying from {args.base}")
        AutoTokenizer.from_pretrained(args.base, token=args.token).save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.out)
    if not getattr(tok, "chat_template", None):
        print(f"[tokenizer] no chat template; using {args.base}'s")
        tok.chat_template = AutoTokenizer.from_pretrained(args.base, token=args.token).chat_template
        tok.save_pretrained(args.out)

    # ---- Verify ------------------------------------------------------------------
    cfg = AutoConfig.from_pretrained(args.out)
    print(f"[ok] {args.out} loads as model_type='{cfg.model_type}', "
          f"architectures={getattr(cfg, 'architectures', None)}")
    print(f"     Run with:  --backend hf-local --model {args.out}")


if __name__ == "__main__":
    main()