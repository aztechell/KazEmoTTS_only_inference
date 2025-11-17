"""Minimal text processing helpers for KazEmoTTS inference."""
from __future__ import annotations
from typing import Iterable, Sequence
import torch
from text import cleaners
from text.symbols import symbols

__all__ = [
    "convert_text",
    "text_to_sequence",
]

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_DEFAULT_CLEANERS: Sequence[str] = ("kazakh_cleaners",)


def text_to_sequence(text: str, cleaner_names: Iterable[str] | None = None) -> list[int]:
    """Convert *text* into a list of symbol identifiers.

    Only the Kazakh cleaner is required for inference, but *cleaner_names* is kept
    for compatibility with the original Tacotron interface.
    """

    names = tuple(cleaner_names) if cleaner_names is not None else _DEFAULT_CLEANERS
    clean_text = _clean_text(text, names)
    return _symbols_to_sequence(clean_text)


def convert_text(string: str) -> tuple[torch.LongTensor, torch.IntTensor]:
    """Return padded text ids and their length tensor for the inference model."""

    text_norm = torch.IntTensor(text_to_sequence(string.lower()))
    text_len = torch.IntTensor([text_norm.size(0)])
    text_padded = torch.zeros(1, text_norm.size(0), dtype=torch.long)
    text_padded[0, : text_norm.size(0)] = text_norm
    return text_padded, text_len


def _clean_text(text: str, cleaner_names: Sequence[str]) -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if cleaner is None:
            raise ValueError(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text


def _symbols_to_sequence(text: str) -> list[int]:
    return [_symbol_to_id[symbol] for symbol in text if _should_keep_symbol(symbol)]


def _should_keep_symbol(symbol: str) -> bool:
    return symbol in _symbol_to_id and symbol not in {"_", "~"}
