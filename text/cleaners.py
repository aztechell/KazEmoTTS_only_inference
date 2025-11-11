"""Kazakh-specific text cleaners used during inference."""

from __future__ import annotations

import re

__all__ = ["kazakh_cleaners"]


_WHITESPACE_RE = re.compile(r"\s+")
_ABBREVIATIONS = [
    (re.compile(r"\bmrs\.", re.IGNORECASE), "misess"),
    (re.compile(r"\bmr\.", re.IGNORECASE), "mister"),
    (re.compile(r"\bdr\.", re.IGNORECASE), "doctor"),
    (re.compile(r"\bst\.", re.IGNORECASE), "saint"),
    (re.compile(r"\bco\.", re.IGNORECASE), "company"),
    (re.compile(r"\bjr\.", re.IGNORECASE), "junior"),
    (re.compile(r"\bmaj\.", re.IGNORECASE), "major"),
    (re.compile(r"\bgen\.", re.IGNORECASE), "general"),
    (re.compile(r"\bdrs\.", re.IGNORECASE), "doctors"),
    (re.compile(r"\brev\.", re.IGNORECASE), "reverend"),
    (re.compile(r"\blt\.", re.IGNORECASE), "lieutenant"),
    (re.compile(r"\bhon\.", re.IGNORECASE), "honorable"),
    (re.compile(r"\bsgt\.", re.IGNORECASE), "sergeant"),
    (re.compile(r"\bcapt\.", re.IGNORECASE), "captain"),
    (re.compile(r"\besq\.", re.IGNORECASE), "esquire"),
    (re.compile(r"\bltd\.", re.IGNORECASE), "limited"),
    (re.compile(r"\bcol\.", re.IGNORECASE), "colonel"),
    (re.compile(r"\bft\.", re.IGNORECASE), "fort"),
]


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text)


def _replace_english_words(text: str) -> str:
    return text.replace("bluetooth не usb", "блютуз не юэсби").replace(
        "mega silk way", "мега силк уэй"
    )


def kazakh_cleaners(text: str) -> str:
    text = text.lower()
    text = _expand_abbreviations(text)
    text = _replace_english_words(text)
    text = _collapse_whitespace(text)
    return text.replace("c", "с").strip()
