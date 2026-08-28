"""
Tests for the binary format signature table (issue #176).

Every signature was checked against ``sample[:12]``, which is wrong twice over.
``ustar`` lives at byte 257 of a tar and so never matched anything, while
``BM``, ``MZ``, ``BZh`` and ``ID3`` are short enough and printable enough to
start an ordinary text file - and a file classified as binary is skipped before
it is decoded, so it never reaches the summarizer or the README.

Covers:
- text files that begin with those letters are read
- the real formats behind them are still detected
- a tar is matched by its own magic, at the offset it lives at
- the offset and confirmation machinery on its own
- an invariant that stops a short printable signature being added unconfirmed
"""

from __future__ import annotations

import bz2
import gzip
import struct
import tarfile
import zipfile

import pytest

from repo2readme.utils.binary import (
    BINARY_SIGNATURES,
    BinarySignature,
    is_binary_content,
)

# Deterministic filler that contains no null byte, so a test that wants the
# signature to be doing the work is not quietly rescued by the null byte rule.
NO_NULLS = bytes(range(1, 256)) * 40


def _write(tmp_path, name: str, data: bytes) -> str:
    target = tmp_path / name
    target.write_bytes(data)
    return str(target)


def _text(tmp_path, name: str, body: str) -> str:
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return str(target)


# ---------------------------------------------------------------------------
# Text that used to be thrown away
# ---------------------------------------------------------------------------


TEXT_CASES = [
    pytest.param("bmw.md", "BMW fleet telemetry parser\n\nSee docs.\n", id="BM"),
    pytest.param("mz.txt", "MZ Consulting style guide\n", id="MZ"),
    pytest.param(
        "id3.py", "ID3_TAG_SIZE = 10\n\n\ndef parse(raw):\n    return raw\n", id="ID3"
    ),
    pytest.param("bzh.md", "BZh compression notes\n", id="BZh"),
    pytest.param("gif.md", "GIF handling\n", id="GIF-prefix"),
    pytest.param("pk.txt", "PK metadata conventions\n", id="PK-prefix"),
]


@pytest.mark.parametrize("name,body", TEXT_CASES)
def test_a_text_file_starting_with_a_signature_prefix_is_read(tmp_path, name, body):
    assert is_binary_content(_text(tmp_path, name, body)) is False


def test_a_source_file_named_after_a_format_is_read(tmp_path):
    body = (
        "ID3v2 tag reader.\n\n"
        "BM_HEADER = 14\n"
        "MZ_OFFSET = 0x3C\n"
    )

    assert is_binary_content(_text(tmp_path, "formats.py", body)) is False


# ---------------------------------------------------------------------------
# The real formats are still detected
# ---------------------------------------------------------------------------


def _bmp() -> bytes:
    pixels = b"\xff\x00\x00\x00\xff\x00\x00\x00"
    dib = struct.pack(
        "<IiiHHIIiiII", 40, 2, 2, 1, 24, 0, len(pixels), 2835, 2835, 0, 0
    )
    body = dib + pixels
    header = struct.pack("<IHHI", 14 + len(body), 0, 0, 14 + len(dib))
    return b"BM" + header + body


def _pe() -> bytes:
    dos = bytearray(b"\x00" * 0x40)
    dos[0:2] = b"MZ"
    dos[2:4] = struct.pack("<H", 0x90)
    dos[0x3C:0x40] = struct.pack("<I", 0x80)
    stub = b"This program cannot be run in DOS mode.\r\r\n$"
    return bytes(dos) + stub.ljust(0x40, b"\x00") + b"PE\x00\x00" + b"\x4c\x01"


def test_a_bitmap_is_detected(tmp_path):
    assert is_binary_content(_write(tmp_path, "a.bmp", _bmp())) is True


def test_a_windows_executable_is_detected(tmp_path):
    assert is_binary_content(_write(tmp_path, "a.exe", _pe())) is True


def test_a_bzip2_archive_is_detected(tmp_path):
    payload = bz2.compress(b"hello world\n" * 200)

    assert is_binary_content(_write(tmp_path, "a.bz2", payload)) is True


def test_a_gzip_archive_is_detected(tmp_path):
    payload = gzip.compress(b"hello world\n" * 200)

    assert is_binary_content(_write(tmp_path, "a.gz", payload)) is True


def test_a_zip_archive_is_detected(tmp_path):
    target = tmp_path / "a.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("member.txt", "hello\n" * 50)

    assert is_binary_content(str(target)) is True


def test_a_png_is_detected(tmp_path):
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + NO_NULLS[:200]

    assert is_binary_content(_write(tmp_path, "a.png", payload)) is True


def test_a_bare_mp3_frame_is_detected(tmp_path):
    payload = b"\xff\xfb\x90\x64" + NO_NULLS[:4000]

    assert is_binary_content(_write(tmp_path, "bare.mp3", payload)) is True


def test_a_tagged_mp3_is_detected(tmp_path):
    # ID3v2.3 header: version, flags, then a syncsafe length whose high bytes
    # are zero for any tag under two megabytes. There is no fixed run of bytes
    # to confirm an "ID3" magic with, which is why the table does not carry
    # one - the null byte rule answers this instead.
    tag = b"ID3" + b"\x03\x00" + b"\x00" + bytes([0, 0, 0x10, 0x00]) + b"\x00" * 2048
    payload = tag + b"\xff\xfb\x90\x64" + NO_NULLS[:1000]

    assert is_binary_content(_write(tmp_path, "tagged.mp3", payload)) is True


# ---------------------------------------------------------------------------
# The tar signature, at the offset it lives at
# ---------------------------------------------------------------------------


def _tar(tmp_path, name: str, fmt) -> str:
    member = tmp_path / "member.txt"
    member.write_text("hello world\n" * 10, encoding="utf-8")
    target = tmp_path / name
    with tarfile.open(target, "w", format=fmt) as archive:
        archive.add(member, arcname="member.txt")
    return str(target)


@pytest.mark.parametrize(
    "fmt", [tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT, tarfile.PAX_FORMAT]
)
def test_a_tar_is_detected(tmp_path, fmt):
    assert is_binary_content(_tar(tmp_path, "a.tar", fmt)) is True


def test_the_tar_magic_is_matched_where_it_actually_sits(tmp_path):
    path = _tar(tmp_path, "a.tar", tarfile.USTAR_FORMAT)
    with open(path, "rb") as handle:
        sample = handle.read(8192)

    assert sample[:12].startswith(b"ustar") is False
    assert sample[257:262] == b"ustar"

    fired = [
        signature.name
        for signature in BINARY_SIGNATURES
        if signature.matches(sample)
    ]
    assert "TAR" in fired


# ---------------------------------------------------------------------------
# The signature type itself
# ---------------------------------------------------------------------------


class TestBinarySignature:
    def test_an_offset_signature_does_not_match_at_the_start(self):
        signature = BinarySignature("test", b"magic", offset=4)

        assert signature.matches(b"magicXXXX") is False
        assert signature.matches(b"XXXXmagic") is True

    def test_a_sample_shorter_than_the_signature_does_not_match(self):
        signature = BinarySignature("test", b"magic", offset=4)

        assert signature.matches(b"XXXX") is False
        assert signature.matches(b"") is False

    def test_confirmation_must_also_match(self):
        signature = BinarySignature("test", b"BM", confirm=(6, b"\x00\x00"))

        assert signature.matches(b"BM" + b"\x01" * 4 + b"\x00\x00") is True
        assert signature.matches(b"BM" + b"\x01" * 4 + b"\x01\x01") is False

    def test_confirmation_beyond_the_sample_does_not_match(self):
        signature = BinarySignature("test", b"BM", confirm=(6, b"\x00\x00"))

        assert signature.matches(b"BM") is False

    def test_every_short_printable_signature_is_confirmed(self):
        # The rule the old table broke: two or three printable characters are
        # not evidence on their own, so anything that short needs corroboration
        # before it is allowed to decide.
        unconfirmed = [
            signature.name
            for signature in BINARY_SIGNATURES
            if signature.confirm is None
            and len(signature.magic) < 4
            and all(0x20 <= byte < 0x7F for byte in signature.magic)
        ]

        assert unconfirmed == []

    def test_signature_names_are_unique(self):
        names = [signature.name for signature in BINARY_SIGNATURES]

        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Nothing else about detection changed
# ---------------------------------------------------------------------------


def test_an_empty_file_is_not_binary(tmp_path):
    assert is_binary_content(_write(tmp_path, "empty.txt", b"")) is False


def test_a_null_byte_still_means_binary(tmp_path):
    assert is_binary_content(_write(tmp_path, "a.bin", b"text\x00more")) is True


def test_a_utf16_file_is_still_text(tmp_path):
    payload = "# Heading\n\nBody text.\n".encode("utf-16")

    assert is_binary_content(_write(tmp_path, "a.md", payload)) is False


def test_an_empty_path_is_rejected():
    with pytest.raises(ValueError):
        is_binary_content("")
