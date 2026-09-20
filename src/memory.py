from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch


def _gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _normalize_cuda_devices(devices: Optional[Iterable[torch.device | int | str]] = None) -> list[torch.device]:
    if not torch.cuda.is_available():
        return []
    if devices is None:
        return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]

    normalized: list[torch.device] = []
    seen: set[int] = set()
    for raw_device in devices:
        device = torch.device(raw_device)
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index in seen:
            continue
        normalized.append(torch.device(f"cuda:{index}"))
        seen.add(index)
    return normalized


def reset_cuda_peak_memory(devices: Optional[Iterable[torch.device | int | str]] = None) -> None:
    for device in _normalize_cuda_devices(devices):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def snapshot_cuda_memory(devices: Optional[Iterable[torch.device | int | str]] = None) -> Dict[str, Any]:
    cuda_devices = _normalize_cuda_devices(devices)
    per_device: Dict[str, Dict[str, float]] = {}
    totals = {
        "allocated_gib": 0.0,
        "reserved_gib": 0.0,
        "max_allocated_gib": 0.0,
        "max_reserved_gib": 0.0,
        "free_gib": 0.0,
        "total_gib": 0.0,
    }
    for device in cuda_devices:
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        entry = {
            "allocated_gib": _gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
            "max_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
            "max_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
            "free_gib": _gib(free_bytes),
            "total_gib": _gib(total_bytes),
        }
        per_device[str(device)] = entry
        for key in totals:
            totals[key] += entry[key]
    return {"cuda": bool(cuda_devices), "per_device": per_device, "total": totals}


def format_cuda_memory_summary(snapshot: Dict[str, Any]) -> str:
    if not snapshot.get("cuda"):
        return "cuda=unavailable"
    total = snapshot.get("total", {})
    return (
        f"alloc={total.get('allocated_gib', 0.0):.2f}GiB "
        f"reserved={total.get('reserved_gib', 0.0):.2f}GiB "
        f"peak_alloc={total.get('max_allocated_gib', 0.0):.2f}GiB "
        f"peak_reserved={total.get('max_reserved_gib', 0.0):.2f}GiB"
    )


def cuda_memory_extra(snapshot: Optional[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Return memory above a model-loaded baseline.

    The baseline should usually be the snapshot taken immediately after model loading.
    `max_*_extra_gib` is the most useful value for context/KV analysis because it removes
    the static model allocation from the peak observed during a generation phase.
    """
    if not snapshot or not baseline:
        return {}
    total = snapshot.get("total", {})
    base_total = baseline.get("total", {})
    return {
        "allocated_extra_gib": float(total.get("allocated_gib", 0.0)) - float(base_total.get("allocated_gib", 0.0)),
        "reserved_extra_gib": float(total.get("reserved_gib", 0.0)) - float(base_total.get("reserved_gib", 0.0)),
        "max_allocated_extra_gib": float(total.get("max_allocated_gib", 0.0)) - float(base_total.get("allocated_gib", 0.0)),
        "max_reserved_extra_gib": float(total.get("max_reserved_gib", 0.0)) - float(base_total.get("reserved_gib", 0.0)),
    }


def format_cuda_memory_extra(extra: Dict[str, float]) -> str:
    if not extra:
        return "extra=unavailable"
    return (
        f"extra_alloc={extra.get('allocated_extra_gib', 0.0):.2f}GiB "
        f"extra_reserved={extra.get('reserved_extra_gib', 0.0):.2f}GiB "
        f"peak_extra_alloc={extra.get('max_allocated_extra_gib', 0.0):.2f}GiB "
        f"peak_extra_reserved={extra.get('max_reserved_extra_gib', 0.0):.2f}GiB"
    )
