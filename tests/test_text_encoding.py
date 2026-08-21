"""
Tests for the shared text decoding chain (issue #127).

Covers:
- the fallback chain itself: BOM sniffing, ordering, truncation tolerance
- the control-character test that keeps latin-1 from calling everything text
- binary detection: UTF-16 and legacy encodings are text, real binaries are not
- the 8 KB sniff boundary no longer splits valid UTF-8 into a "binary" file
- load_file_content: what it decodes, what it refuses, and where it reports it
"""

from __future__ import annotations

import codecs
import logging

import pytest

from repo2readme.loaders.traversal.stages import load_file_content
from repo2readme.utils.binary import is_binary_content
from repo2readme.utils.text_encoding import (
    BOM_ENCODINGS,
    FALLBACK_ENCODINGS,
    MAX_CONTROL_RATIO,
    control_ratio,
    decode_bytes,
    describe_chain,
    encoding_from_bom,
    looks_like_text,
)

# ---------------------------------------------------------------------------
# BOM sniffing
# ---------------------------------------------------------------------------


class TestEncodingFromBom:
    def test_no_bom(self):
        assert encoding_from_bom(b"print('hi')\n") is None

    def test_empty(self):
        assert encoding_from_bom(b"") is None

    def test_utf8_bom(self):
        assert encoding_from_bom(codecs.BOM_UTF8 + b"x") == "utf-8-sig"

    @pytest.mark.parametrize("bom", [codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE])
    def test_utf16_bom(self, bom):
        assert encoding_from_bom(bom + b"x\x00") == "utf-16"

    @pytest.mark.parametrize("bom", [codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE])
    def test_utf32_bom(self, bom):
        assert encoding_from_bom(bom + b"x\x00\x00\x00") == "utf-32"

    def test_utf32_le_wins_over_utf16_le(self):
        """FF FE 00 00 starts with the UTF-16 LE mark; the longer match wins."""
        assert encoding_from_bom(codecs.BOM_UTF32_LE) == "utf-32"

    def test_longest_bom_is_checked_first(self):
        lengths = [len(bom) for bom, _ in BOM_ENCODINGS]
        utf32 = [n for n, (_, enc) in zip(lengths, BOM_ENCODINGS) if enc == "utf-32"]
        utf16 = [n for n, (_, enc) in zip(lengths, BOM_ENCODINGS) if enc == "utf-16"]
        assert min(utf32) > max(utf16)


# ---------------------------------------------------------------------------
# decode_bytes
# ---------------------------------------------------------------------------


class TestDecodeBytes:
    def test_empty_input(self):
        result = decode_bytes(b"")
        assert result is not None
        assert result.text == ""

    def test_plain_ascii(self):
        result = decode_bytes(b"print('hi')\n")
        assert result.text == "print('hi')\n"
        assert result.encoding == "utf-8-sig"
        assert result.from_bom is False

    def test_utf8_with_non_ascii(self):
        result = decode_bytes("# café\n".encode())
        assert result.text == "# café\n"
        assert result.encoding == "utf-8-sig"

    def test_utf8_bom_is_consumed_not_kept(self):
        """A BOM must not survive into the content as a leading \\ufeff."""
        result = decode_bytes(codecs.BOM_UTF8 + b"print('hi')\n")
        assert result.text == "print('hi')\n"
        assert not result.text.startswith("﻿")
        assert result.from_bom is True

    def test_utf16(self):
        result = decode_bytes("# café\nprint('hi')\n".encode("utf-16"))
        assert result.text == "# café\nprint('hi')\n"
        assert result.encoding == "utf-16"
        assert result.from_bom is True

    def test_utf32(self):
        result = decode_bytes("x = 1\n".encode("utf-32"))
        assert result.text == "x = 1\n"
        assert result.encoding == "utf-32"

    def test_cp1252_before_latin1(self):
        """0x93/0x94 are smart quotes in cp1252 and undefined controls in latin-1."""
        result = decode_bytes(b"# \x93quoted\x94\n")
        assert result.encoding == "cp1252"
        assert result.text == "# “quoted”\n"

    def test_latin1_terminates_the_chain(self):
        """0x81 has no cp1252 mapping, so latin-1 is what accepts it."""
        result = decode_bytes(b"# caf\x81\n")
        assert result.encoding == "latin-1"

    def test_returns_none_when_the_chain_cannot_decode(self):
        assert decode_bytes(b"\xff\xfe\x00", encodings=("utf-8",)) is None

    def test_truncated_multibyte_character_rejected_by_default(self):
        truncated = "café".encode()[:-1]
        assert decode_bytes(truncated, encodings=("utf-8",)) is None

    def test_truncated_multibyte_character_allowed_when_asked(self):
        truncated = "café".encode()[:-1]
        result = decode_bytes(truncated, encodings=("utf-8",), allow_truncated=True)
        assert result is not None
        assert result.text == "caf"

    def test_truncation_tolerance_does_not_excuse_real_invalid_bytes(self):
        """Trimming the tail must not rescue bytes that are invalid up front."""
        raw = b"a" * 100 + b"\xc3\x28" + b"a" * 100
        assert decode_bytes(raw, encodings=("utf-8",), allow_truncated=True) is None

    def test_chain_order_is_most_to_least_trustworthy(self):
        assert FALLBACK_ENCODINGS[0] == "utf-8-sig"
        assert FALLBACK_ENCODINGS[-1] == "latin-1"
        assert FALLBACK_ENCODINGS.index("cp1252") < FALLBACK_ENCODINGS.index("latin-1")

    def test_describe_chain_lists_every_encoding(self):
        described = describe_chain()
        for encoding in FALLBACK_ENCODINGS:
            assert encoding in described


# ---------------------------------------------------------------------------
# The control-character test
# ---------------------------------------------------------------------------


class TestLooksLikeText:
    def test_source_code(self):
        assert looks_like_text("def f():\n\treturn 1\n") is True

    def test_empty(self):
        assert looks_like_text("") is True

    def test_tabs_newlines_and_escapes_are_not_control_soup(self):
        assert control_ratio("a\tb\nc\r\nd\x1b[0m") == 0.0

    def test_null_byte(self):
        assert looks_like_text("ok\x00ok") is False

    def test_control_heavy_string(self):
        assert looks_like_text("\x01\x02\x03\x04\x05" * 20) is False

    def test_ratio_is_the_threshold(self):
        mostly_text = "a" * 100 + "\x01"
        assert control_ratio(mostly_text) < MAX_CONTROL_RATIO
        assert looks_like_text(mostly_text) is True

    def test_c1_range_counts_as_control(self):
        assert control_ratio("\x80\x9f") == 1.0


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------


class TestBinaryDetectionAcceptsTextEncodings:
    """The cases from issue #127 that were reported to the user as binary."""

    def test_utf16_source_file_is_text(self, tmp_path):
        path = tmp_path / "app.py"
        path.write_bytes("# café\nprint('hi')\n".encode("utf-16"))
        assert is_binary_content(str(path)) is False

    def test_utf16_be_source_file_is_text(self, tmp_path):
        path = tmp_path / "app.py"
        path.write_bytes(codecs.BOM_UTF16_BE + "x = 1\n".encode("utf-16-be"))
        assert is_binary_content(str(path)) is False

    def test_latin1_source_file_is_text(self, tmp_path):
        path = tmp_path / "app.py"
        path.write_bytes("# caf\xe9\nprint('hi')\n".encode("latin-1"))
        assert is_binary_content(str(path)) is False

    def test_cp1252_source_file_is_text(self, tmp_path):
        path = tmp_path / "app.py"
        path.write_bytes(b"# smart \x93quotes\x94\nprint('hi')\n")
        assert is_binary_content(str(path)) is False

    def test_valid_utf8_split_by_the_sample_boundary_is_text(self, tmp_path):
        """A multi-byte character straddling byte 8192 is not evidence of anything."""
        path = tmp_path / "long.py"
        raw = b"#" + b"a" * 8190 + "é".encode() + b"\nprint(1)\n"
        assert raw[8191:8193] == "é".encode()
        path.write_bytes(raw)

        assert raw.decode("utf-8")  # the file really is valid UTF-8
        assert is_binary_content(str(path)) is False

    def test_the_two_stages_agree(self, tmp_path):
        """Whatever the sniff calls text, the loader must be able to read."""
        samples = {
            "utf16.py": "# café\n".encode("utf-16"),
            "latin1.py": "# caf\xe9\n".encode("latin-1"),
            "cp1252.py": b"# \x93hi\x94\n",
            "bom.py": codecs.BOM_UTF8 + b"print('hi')\n",
            "plain.py": b"print('hi')\n",
        }
        for name, raw in samples.items():
            path = tmp_path / name
            path.write_bytes(raw)
            if is_binary_content(str(path)):
                continue
            content, error = load_file_content(str(path))
            assert error is None, f"{name} sniffed as text but failed to load"
            assert content is not None


class TestBinaryDetectionStillCatchesBinaries:
    """The regression side: loosening the decode must not admit real binaries."""

    def test_png(self, tmp_path):
        path = tmp_path / "image.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)
        assert is_binary_content(str(path)) is True

    def test_zip(self, tmp_path):
        path = tmp_path / "archive.zip"
        path.write_bytes(b"PK\x03\x04" + bytes(range(256)) * 4)
        assert is_binary_content(str(path)) is True

    def test_elf(self, tmp_path):
        path = tmp_path / "a.out"
        path.write_bytes(b"\x7fELF" + bytes(range(256)) * 4)
        assert is_binary_content(str(path)) is True

    def test_null_bytes_without_a_signature(self, tmp_path):
        path = tmp_path / "blob.dat"
        path.write_bytes(b"header" + b"\x00" * 100 + b"trailer")
        assert is_binary_content(str(path)) is True

    def test_control_byte_soup_without_null_bytes(self, tmp_path):
        """No null byte, no signature - only the control ratio catches this."""
        path = tmp_path / "blob.dat"
        path.write_bytes(bytes(range(1, 32)) * 200)
        assert is_binary_content(str(path)) is True

    def test_bom_less_utf16_is_still_binary(self, tmp_path):
        """Documented limitation: without a BOM this is indistinguishable."""
        path = tmp_path / "app.py"
        path.write_bytes("print('hi')\n".encode("utf-16-le"))
        assert is_binary_content(str(path)) is True

    def test_empty_file_is_not_binary(self, tmp_path):
        path = tmp_path / "empty.py"
        path.write_bytes(b"")
        assert is_binary_content(str(path)) is False

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError):
            is_binary_content("")


# ---------------------------------------------------------------------------
# load_file_content
# ---------------------------------------------------------------------------


class TestLoadFileContent:
    def test_plain_utf8(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_text("print('hi')\n", encoding="utf-8")
        assert load_file_content(str(path)) == ("print('hi')\n", None)

    def test_utf16_is_decoded_instead_of_dropped(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_bytes("# café\nprint('hi')\n".encode("utf-16"))
        content, error = load_file_content(str(path))
        assert error is None
        assert content == "# café\nprint('hi')\n"

    def test_latin1_is_decoded_instead_of_dropped(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_bytes("# caf\xe9\nprint('hi')\n".encode("latin-1"))
        content, error = load_file_content(str(path))
        assert error is None
        assert content == "# café\nprint('hi')\n"

    def test_bom_is_stripped_from_the_content(self, tmp_path):
        """The BOM used to become the first character of the summarized file."""
        path = tmp_path / "a.py"
        path.write_bytes(codecs.BOM_UTF8 + b"print('hi')\n")
        content, error = load_file_content(str(path))
        assert error is None
        assert content == "print('hi')\n"
        assert not content.startswith("﻿")

    def test_legacy_bytes_past_the_sniff_window(self, tmp_path):
        """Valid UTF-8 for 8 KB, legacy afterwards: used to be encoding_error."""
        path = tmp_path / "late.py"
        path.write_bytes(b"x = 1\n" * 2000 + "# caf\xe9\n".encode("latin-1"))
        content, error = load_file_content(str(path))
        assert error is None
        assert content.endswith("# café\n")

    def test_missing_file_reports_a_permission_error(self, tmp_path):
        content, error = load_file_content(str(tmp_path / "nope.py"))
        assert content is None
        assert error.startswith("permission_error")

    def test_directory_reports_an_error_rather_than_raising(self, tmp_path):
        content, error = load_file_content(str(tmp_path))
        assert content is None
        assert error is not None

    def test_errors_are_logged_not_printed(self, tmp_path, capsys, caplog):
        """These run underneath a rich progress bar; stdout belongs to it."""
        with caplog.at_level(logging.WARNING):
            load_file_content(str(tmp_path / "nope.py"))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "[ERROR]" not in captured.out
        assert any("nope.py" in record.getMessage() for record in caplog.records)

    def test_successful_read_logs_nothing_at_warning_level(self, tmp_path, caplog):
        path = tmp_path / "a.py"
        path.write_text("print('hi')\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            load_file_content(str(path))
        assert caplog.records == []

    def test_chosen_encoding_is_reported_at_debug_level(self, tmp_path, caplog):
        path = tmp_path / "a.py"
        path.write_bytes("# café\n".encode("utf-16"))
        with caplog.at_level(logging.DEBUG):
            load_file_content(str(path))
        assert any("utf-16" in record.getMessage() for record in caplog.records)
