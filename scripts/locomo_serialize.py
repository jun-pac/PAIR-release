#!/usr/bin/env python
"""Deterministic full-conversation serializer for LoCoMo (snap-research/locomo, data/locomo10.json).

One conversation -> ONE chronological context string shared by ALL its QA questions (fixed shared-context setting).
Preserves session order (numeric, not lexical), session date/time, speaker names, dialogue order, dialogue text.
Image turns (no images in the release) -> the provided BLIP caption inserted at the turn position, MARKED as a caption.
Also returns dia_id -> (session_idx, turn_idx) so evidence-turn stats can be computed. No model, no GPU.
"""
import re, hashlib


def _session_keys(conv):
    """session_N keys sorted by the integer N (session_2 before session_10)."""
    ks = [k for k in conv if re.fullmatch(r"session_\d+", k)]
    return sorted(ks, key=lambda k: int(k.split("_")[1]))


def serialize_conversation(conv):
    """Return (context_text, dia_positions, n_image_turns).
    dia_positions: dia_id -> {'session': N, 'turn_index_in_session': i, 'char_offset': o, 'is_image': bool}."""
    lines = []
    dia_positions = {}
    n_img = 0
    for sk in _session_keys(conv):
        n = int(sk.split("_")[1])
        dt = conv.get(f"session_{n}_date_time", "")
        lines.append(f"[Session {n} | {dt}]")
        turns = conv[sk] or []
        for i, t in enumerate(turns):
            spk = t.get("speaker", "?")
            txt = (t.get("text") or "").strip()
            cap = t.get("blip_caption")
            is_img = bool(cap)
            if is_img:
                n_img += 1
                txt = f"{txt} [shared an image — caption: {cap.strip()}]"
            off = sum(len(x) + 1 for x in lines)  # char offset where this line starts
            lines.append(f"{spk}: {txt}")
            did = t.get("dia_id")
            if did:
                dia_positions[did] = {"session": n, "turn_index_in_session": i,
                                      "char_offset": off, "is_image": is_img}
        lines.append("")  # blank line between sessions
    text = "\n".join(lines).strip() + "\n"
    return text, dia_positions, n_img


def context_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import json, sys
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/work/hdd/myproject/anon/locomo/locomo10.json"))
    c = d[0]
    text, pos, nimg = serialize_conversation(c["conversation"])
    print(f"sample_id={c['sample_id']}  chars={len(text)}  image_turns={nimg}  hash={context_hash(text)[:16]}")
    print(text[:1200])
