import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import re
from typing import Dict, List, Optional, Any, Tuple
from transformers import GenerationConfig


def _build_bnb_config(quantize: Optional[str]) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig for 4bit or 8bit quantization, or None."""
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantize is not None:
        raise ValueError(f"Unknown quantize option '{quantize}'. Use '4bit', '8bit', or None.")
    return None


def load_local_director_model(
    model_path: str,
    quantize: Optional[str] = None,   # "4bit", "8bit", or None
):
    bnb_config = _build_bnb_config(quantize)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=bnb_config,   # None is silently ignored by HF
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return model, tokenizer

def load_local_director_pipeline(
    model_path: str,
    quantize: Optional[str] = None,
    gpus: Optional[List[int]] = None,   # e.g. [0, 1] or [0] or None
    lora_path: Optional[str] = None,
    max_new_tokens: int = 512,
):
    bnb_config = _build_bnb_config(quantize)

    # Build device_map
    if gpus and len(gpus) == 2:
        # Manual split across two GPUs — avoids Blackwell "auto" bug
        # Load model first in meta device to inspect layer count, then assign
        from accelerate import infer_auto_device_map
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        n_layers = config.num_hidden_layers

        split = n_layers // 2
        device_map = {}
        device_map["model.embed_tokens"] = f"cuda:{gpus[0]}"
        device_map["model.norm"] = f"cuda:{gpus[1]}"
        device_map["lm_head"] = f"cuda:{gpus[1]}"
        for i in range(n_layers):
            gpu = gpus[0] if i < split else gpus[1]
            device_map[f"model.layers.{i}"] = f"cuda:{gpu}"

    elif gpus and len(gpus) == 1:
        device_map = f"cuda:{gpus[0]}"
    else:
        device_map = "cuda:0"   # safe single-GPU fallback

    pipe = pipeline(
            "text-generation",
            model=model_path,
            device_map=device_map,
            dtype=torch.bfloat16,
            max_new_tokens=max_new_tokens,
            model_kwargs={"quantization_config": bnb_config} if bnb_config else {},
        )
    pipe.tokenizer.padding_side = "left"

    if lora_path is not None:
        from peft import PeftModel
        pipe.model = PeftModel.from_pretrained(
            pipe.model,
            lora_path,
            is_trainable=False,
        )
        print("before lora merging")
        pipe.model = pipe.model.merge_and_unload()   # optional: merge weights for faster inference
        print("after  lora merging", pipe.model )
        print(f"  LoRA loaded and merged from {lora_path}")
  
    return pipe, None


# # single GPU (original behavior)
# pipe, _ = load_local_director_pipeline("Qwen/Qwen2.5-7B", quantize="4bit")

# # manual split across GPU 0 and 1
# pipe, _ = load_local_director_pipeline("Qwen/Qwen2.5-7B", quantize="4bit", gpus=[0, 1])

