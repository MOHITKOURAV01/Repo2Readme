"""One shared notion of "decodable text" for the traversal pipeline.

Two stages used to decide independently whether a file was readable, and they
disagreed. :func:`repo2readme.utils.binary.is_binary_content` sniffed an 8 KB
prefix and called anything that was not valid UTF-8 binary;
``load_file_content`` then decoded the whole file as UTF-8 and reported
``encoding_error`` when that failed. Between them they dropped every UTF-16 and
cp1252 file in a repository - as ``binary_file`` when the offending byte was
inside the sample, as ``encoding_error`` when it was past it - and they dropped
valid UTF-8 files whose 8192nd byte happened to fall in the middle of a
multi-byte character.

Both stages now ask this module instead, so whatever the sniff calls text, the
loader can read.

The chain is deliberately short and ordered from most to least trustworthy:

``utf-8-sig``
    UTF-8, and identical to it except that a byte order mark is consumed rather
    than becoming the first character of the file.
BOM-declared encoding
    A UTF-16 or UTF-32 byte order mark is an explicit statement of encoding.
``cp1252`` then ``latin-1``
    Legacy single-byte encodings. ``cp1252`` first because it is what Windows
    editors actually wrote; ``latin-1`` cannot fail, so it terminates the chain.

Because ``latin-1`` always succeeds, "is this text?" cannot be answered by
decoding alone - :func:`looks_like_text` applies a control-byte test to the
result, which is what still separates a legacy-encoded source file from a
binary blob.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass

# Encodings tried, in order, when there is no byte order mark to go on.
FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")

# Byte order marks, longest first: the UTF-32 marks start with the UTF-16 ones
# (``FF FE 00 00`` begins with ``FF FE``), so a shorter match must never win.
BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# Control characters that appear in ordinary text files.
_ALLOWED_CONTROLS = frozenset("\t\n\r\f\v\x1b")

# Above this share of control characters the decode is assumed to have produced
# mojibake from binary input rather than text. ``latin-1`` maps all 256 byte
# values, so without this every file on disk would be "text".
MAX_CONTROL_RATIO = 0.05

# A decode is only judged on a bounded prefix; a long file does not need to be
# scanned twice to answer a yes/no question.
TEXT_CHECK_LIMIT = 4096


@dataclass(frozen=True)
class DecodedText:
    """The result of decoding a byte string.

    Attributes
    ----------
    text:
        The decoded content.
    encoding:
        The codec that produced it, as passed to :func:`bytes.decode`.
    from_bom:
        Whether the codec was chosen because the input carried a byte order
        mark, rather than by trying the fallback chain.
    """

    text: str
    encoding: str
    from_bom: bool = False


def encoding_from_bom(raw: bytes) -> str | None:
    """Return the encoding declared by a leading byte order mark, if any."""
    for bom, encoding in BOM_ENCODINGS:
        if raw.startswith(bom):
            return encoding
    return None


def _decode_strict(raw: bytes, encoding: str) -> str | None:
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return None


def _decode_ignoring_truncation(raw: bytes, encoding: str) -> str | None:
    """Decode ``raw``, tolerating a multi-byte character cut off at the end.

    A fixed-size sample of a valid UTF-8 file regularly ends mid-character.
    Dropping up to the maximum character width from the tail is enough to tell
    a truncated sample from genuinely invalid bytes: if the remainder still
    fails to decode, the problem was not the boundary.
    """
    for trim in range(5):
        candidate = raw[: len(raw) - trim] if trim else raw
        if not candidate:
            return ""
        decoded = _decode_strict(candidate, encoding)
        if decoded is not None:
            return decoded
    return None


def control_ratio(text: str, limit: int = TEXT_CHECK_LIMIT) -> float:
    """Share of characters in ``text`` that are control characters.

    Tab, newline, carriage return, form feed, vertical tab and escape are
    excluded: they are ordinary in source files, logs and terminal captures.
    """
    sample = text[:limit] if limit else text
    if not sample:
        return 0.0

    controls = sum(
        1
        for char in sample
        if char not in _ALLOWED_CONTROLS
        and (ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F)
    )
    return controls / len(sample)


def looks_like_text(text: str, max_control_ratio: float = MAX_CONTROL_RATIO) -> bool:
    """Whether a decoded string plausibly came from a text file.

    Only meaningful for a decode that cannot fail. ``latin-1`` turns any byte
    sequence into a string, so the question "did that decode mean anything?"
    has to be answered separately, and the density of control characters is the
    cheapest signal that answers it.
    """
    if not text:
        return True
    if "\x00" in text[:TEXT_CHECK_LIMIT]:
        return False
    return control_ratio(text) <= max_control_ratio


def decode_bytes(
    raw: bytes,
    *,
    encodings: tuple[str, ...] = FALLBACK_ENCODINGS,
    allow_truncated: bool = False,
) -> DecodedText | None:
    """Decode ``raw`` with the first encoding that accepts it.

    Parameters
    ----------
    raw:
        The bytes to decode.
    encodings:
        The fallback chain, used when there is no byte order mark.
    allow_truncated:
        Set when ``raw`` is a fixed-size prefix of a larger file, so a
        multi-byte character split by the sample boundary is not mistaken for
        invalid input.

    Returns
    -------
    DecodedText | None
        ``None`` only when every candidate encoding refused the input, which
        cannot happen while ``latin-1`` is in the chain.
    """
    if not raw:
        return DecodedText(text="", encoding="utf-8")

    decode = _decode_ignoring_truncation if allow_truncated else _decode_strict

    bom_encoding = encoding_from_bom(raw)
    if bom_encoding is not None:
        decoded = decode(raw, bom_encoding)
        if decoded is not None:
            return DecodedText(text=decoded, encoding=bom_encoding, from_bom=True)

    for encoding in encodings:
        decoded = decode(raw, encoding)
        if decoded is not None:
            return DecodedText(text=decoded, encoding=encoding)

    return None


def describe_chain(encodings: tuple[str, ...] = FALLBACK_ENCODINGS) -> str:
    """Human readable form of a fallback chain, for error messages."""
    return ", ".join(encodings)
