from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


_SUPPORTED_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential", "cpu"}


def normalize_device_map(device_map: Optional[str], gpu_count: int) -> str:
    if device_map is None:
        return "auto"
    name = str(device_map).strip()
    if name in _SUPPORTED_DEVICE_MAPS:
        return name
    if name.startswith("cuda:"):
        try:
            idx = int(name.split(":", 1)[1])
        except ValueError:
            return name
        if idx >= gpu_count:
            print(f"Requested {name} but only {gpu_count} GPU(s); falling back to cuda:0.")
            return "cuda:0"
    return name


def _iter_module_devices(module) -> Iterable[torch.device]:
    if module is None:
        return ()
    try:
        params = list(module.parameters())
    except Exception:  # noqa: BLE001
        return ()
    return [param.device for param in params]


def _device_from_map_value(value) -> Optional[torch.device]:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    if isinstance(value, str):
        if value == "disk":
            return None
        return torch.device(value)
    return None


def get_model_cuda_devices(model: PreTrainedModel) -> list[torch.device]:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        devices = []
        seen = set()
        for value in hf_device_map.values():
            device = _device_from_map_value(value)
            if device is None or device.type != "cuda" or device in seen:
                continue
            devices.append(device)
            seen.add(device)
        if devices:
            return devices

    devices = []
    seen = set()
    modules = [model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None]
    if hasattr(model, "get_output_embeddings"):
        modules.append(model.get_output_embeddings())
    modules.append(model)
    for module in modules:
        for device in _iter_module_devices(module):
            if device.type != "cuda" or device in seen:
                continue
            devices.append(device)
            seen.add(device)
    return devices


def get_model_input_device(model: PreTrainedModel) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        devices = list(_iter_module_devices(model.get_input_embeddings()))
        if devices:
            return devices[0]
    cuda_devices = get_model_cuda_devices(model)
    if cuda_devices:
        return cuda_devices[0]
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    return next(model.parameters()).device


def get_model_output_device(model: PreTrainedModel) -> torch.device:
    if hasattr(model, "get_output_embeddings"):
        devices = list(_iter_module_devices(model.get_output_embeddings()))
        if devices:
            return devices[0]
    cuda_devices = get_model_cuda_devices(model)
    if cuda_devices:
        return cuda_devices[-1]
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    return next(model.parameters()).device


def synchronize_model(model: PreTrainedModel) -> None:
    if not torch.cuda.is_available():
        return
    devices = get_model_cuda_devices(model)
    if not devices:
        device = getattr(model, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device]
    for device in devices:
        torch.cuda.synchronize(device)


def describe_device_map(model: PreTrainedModel) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict):
        return str(get_model_input_device(model))
    counts = {}
    for value in hf_device_map.values():
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{device} ({count})" for device, count in sorted(counts.items()))


def load_causal_lm(
    model_name: str,
    *,
    device_map="auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
    max_memory: Optional[dict] = None,
    quantization_config=None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM with tokenizer configured for generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "device_map": device_map,
        "dtype": torch_dtype,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "max_memory": max_memory,
    }
    # ★ Use FlashAttention-2 when available — huge for long-context prefill AND per-token decode over a long
    # KV cache. Default was sdpa/eager → manual & generate both crawled on summarization/long contexts.
    # Checked at LOAD time on the GPU node (is_flash_attn_2_available() is False on the CPU login node).
    import os as _os
    _attn = _os.environ.get("ATTN_IMPL")
    if not _attn:
        try:
            from transformers.utils import is_flash_attn_2_available
            _attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        except Exception:
            _attn = "sdpa"
    model_kwargs["attn_implementation"] = _attn
    # ★ YaRN rope-scaling (env-gated): extends a base model's context past its native window so a SMALL base
    # SLM (e.g. Qwen2.5-3B, native 32k tok) can read >32k-token contexts — preserving the BIG 14B-3B gap at
    # long context (there is no 3B-1M variant; the -1M path forces the weak 14B-7B gap). Only activates when
    # ROPE_YARN_FACTOR is set. NOT position-local (corrected 2026-09-11): HF's yarn rope_type rewrites inv_freq
    # (low-frequency dims interpolated by the factor) AND multiplies cos/sin by attention_scaling
    # 0.1*ln(factor)+1 (=1.1386 at 4, i.e. q·k logits x1.30) at EVERY position, so the query-only LM's
    # short input is changed too. Set it on both models or on neither; the context axis sets both.
    _yarn = _os.environ.get("ROPE_YARN_FACTOR")
    if _yarn:
        model_kwargs["rope_scaling"] = {
            "rope_type": "yarn", "factor": float(_yarn), "original_max_position_embeddings": 32768,
        }
        print(f"[Info] YaRN rope_scaling factor={_yarn} -> ~{int(32768*float(_yarn))} tok window")
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    # ★ TP_PLAN=auto — TRUE TENSOR PARALLELISM (2026-09-02). Splits every attention and MLP matmul
    # across the ranks (Qwen2 ships base_model_tp_plan: q/k/v/gate/up colwise, o/down rowwise) with
    # an all-reduce per block, so BOTH cards compute every token instead of taking turns. This is
    # NOT device_map="balanced": that is naive LAYER sharding, which gives the same memory headroom
    # but runs the cards SEQUENTIALLY -- measured on LooGLE it reached batch 12 against one card's 3
    # and still lost, 0.1121 answers/s against two replicas' 0.3326 (x0.337), because a decode step
    # now costs one card's compute plus a cross-card hop. TP is the version that can actually win,
    # and it exists so the teacher is compared at its best two-card option rather than a strawman.
    # Requires a torchrun process group; device_map must be left unset so TP owns placement.
    if _os.environ.get("TP_PLAN"):
        model_kwargs["tp_plan"] = _os.environ["TP_PLAN"]
        model_kwargs.pop("device_map", None)
        model_kwargs.pop("max_memory", None)
        # Each rank must own its card BEFORE the weights are materialised. Without this both ranks
        # allocate against the current device (cuda:0) while the TP mesh believes they are on
        # different ones, and the load dies in native code with `free(): double free detected in
        # tcache 2` / SIGABRT rather than a Python error (job 3074000).
        _lr = int(_os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(_lr)
        print(f"[Info] TP_PLAN={model_kwargs['tp_plan']} on local_rank {_lr} "
              f"(cuda:{_lr}); device_map ignored — TP owns placement", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as _e:  # flash kernel missing/incompatible for this arch → fall back to sdpa
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            print(f"[Info] flash_attention_2 load failed ({type(_e).__name__}); falling back to sdpa")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise
    print(f"[Info] attn_implementation={model_kwargs['attn_implementation']}")
    model.eval()
    if max_memory:
        print(f"[Info] Loading {model_name} with device_map={device_map} max_memory={max_memory}")
    print(f"[Info] Loaded {model_name} with device_map={device_map} -> {describe_device_map(model)}")
    return model, tokenizer


# Opt-in chat-template wrapping. Default False keeps prepare_inputs byte-identical
# to the historical raw-prompt behavior; only an explicit set_use_chat_template(True)
# (e.g. via the --use-chat-template CLI flag) enables the chat-template branch.
_USE_CHAT_TEMPLATE = False


def set_use_chat_template(v: bool) -> None:
    global _USE_CHAT_TEMPLATE
    _USE_CHAT_TEMPLATE = bool(v)


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int = 150000,
    return_tensors: str = "pt",
) -> dict:
    """Tokenize text for generation with consistent truncation/padding rules."""
    if _USE_CHAT_TEMPLATE and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Tokenizer without a usable template: fall back to the raw text.
            pass
        else:
            # The chat template already injects BOS/special tokens; adding them
            # again here would corrupt the prompt.
            return tokenizer(
                text,
                return_tensors=return_tensors,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )
    return tokenizer(
        text,
        return_tensors=return_tensors,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
