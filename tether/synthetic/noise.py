"""Noise operators used by the synthetic generator. All take an explicit ``random.Random``."""

from __future__ import annotations

import random
import re

from tether.synthetic.pools import DIRECTIONALS, ORG_ABBREVIATIONS, STREET_TYPES

_KEYBOARD_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm",
    "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy", "u": "yhji",
    "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}


def typo(text: str, rng: random.Random) -> str:
    """Introduce one realistic character-level error (substitute, transpose, drop, double)."""
    if len(text) < 3:
        return text
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not letters:
        return text
    i = rng.choice(letters)
    op = rng.choice(("substitute", "transpose", "drop", "double"))
    chars = list(text)
    if op == "substitute":
        low = chars[i].lower()
        repl = rng.choice(_KEYBOARD_NEIGHBOURS.get(low, "e"))
        chars[i] = repl.upper() if chars[i].isupper() else repl
    elif op == "transpose" and i + 1 < len(chars) and chars[i + 1].isalpha():
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "drop":
        del chars[i]
    else:
        chars.insert(i, chars[i])
    return "".join(chars)


def vary_case(text: str, rng: random.Random) -> str:
    """Randomly upper-case, lower-case or leave a string."""
    return rng.choice((text, text.upper(), text.lower(), text))


def abbreviate_org(name: str, rng: random.Random, max_subs: int = 2) -> str:
    """Replace up to ``max_subs`` whole words with common abbreviations."""
    words = name.split(" ")
    idx = [i for i, w in enumerate(words) if w in ORG_ABBREVIATIONS]
    rng.shuffle(idx)
    for i in idx[:max_subs]:
        words[i] = ORG_ABBREVIATIONS[words[i]]
    return " ".join(words)


def reformat_street(street: str, rng: random.Random) -> str:
    """Toggle street-type/directional abbreviations, add periods, drop unit, vary case."""
    out = street
    for full, abbr in STREET_TYPES + DIRECTIONALS:
        if re.search(rf"\b{full}\b", out):
            out = re.sub(rf"\b{full}\b", rng.choice((abbr, abbr + ".", full)), out)
        elif re.search(rf"\b{abbr}\b", out):
            out = re.sub(rf"\b{abbr}\b", rng.choice((full, abbr + ".", abbr)), out)
    out = re.sub(r"\bSuite\b", rng.choice(("Suite", "Ste", "Ste.", "#")), out)
    if rng.random() < 0.3:
        out = re.sub(r",?\s*(Suite|Ste\.?|#)\s*\w+$", "", out)
    return vary_case(out, rng)


def reformat_phone(digits: str, rng: random.Random) -> str:
    """Render 10 digits in one of several common formats."""
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice((f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}", digits, f"1-{a}-{b}-{c}", f"+1 {a} {b} {c}"))
