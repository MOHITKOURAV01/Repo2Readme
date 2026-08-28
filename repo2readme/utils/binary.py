"""
Binary file detection utilities.

Provides content-based detection of binary files to allow the traversal
pipeline to skip unsupported content before attempting text decoding,
metadata extraction, or language detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from repo2readme.utils.text_encoding import (
    decode_bytes,
    encoding_from_bom,
    looks_like_text,
)

# Bounded sample size for binary detection (8 KB).
# This matches the sample size used in repo2readme.utils.detect_language.
_MAX_SAMPLE_SIZE = 8192

@dataclass(frozen=True)
class BinarySignature:
    """A byte pattern that identifies a binary format.

    ``magic`` sits at ``offset``, which is not always zero: a tar's ``ustar``
    lives at byte 257, in the middle of the first member's header, so looking
    for it at the start of the file never found one.

    ``confirm`` is a second, separately anchored run of bytes that must also
    match. It exists because a signature made only of printable characters is
    not evidence on its own - ``BM``, ``MZ`` and ``BZh`` are all things an
    ordinary text file can begin with - while every format that starts that way
    has something further into its header that is. A signature that needs no
    corroboration leaves it as ``None``.
    """

    name: str
    magic: bytes
    offset: int = 0
    confirm: tuple[int, bytes] | None = None

    def matches(self, sample: bytes) -> bool:
        if not self._at(sample, self.offset, self.magic):
            return False
        if self.confirm is None:
            return True
        return self._at(sample, *self.confirm)

    @staticmethod
    def _at(sample: bytes, offset: int, expected: bytes) -> bool:
        end = offset + len(expected)
        return len(sample) >= end and sample[offset:end] == expected


# Known binary file signatures. A file whose sample matches any of them is
# binary without further inspection, so each one has to be strong enough to
# carry that on its own.
BINARY_SIGNATURES: tuple[BinarySignature, ...] = (
    # Images
    BinarySignature("PNG", b"\x89PNG\r\n\x1a\n"),
    BinarySignature("JPEG", b"\xff\xd8\xff"),
    BinarySignature("GIF87a", b"GIF87a"),
    BinarySignature("GIF89a", b"GIF89a"),
    # "BM" alone is two letters. Bytes 6-9 are the BMP header's two reserved
    # 16-bit fields, which every writer leaves zero.
    BinarySignature("BMP", b"BM", confirm=(6, b"\x00\x00\x00\x00")),
    BinarySignature("ICO", b"\x00\x00\x01\x00"),
    # Documents
    BinarySignature("PDF", b"%PDF"),
    # Archives / compressed
    BinarySignature("ZIP", b"PK\x03\x04"),  # also JAR, DOCX, XLSX, PPTX, APK
    BinarySignature("RAR4", b"Rar!\x1a\x07\x00"),
    BinarySignature("RAR5", b"Rar!\x1a\x07\x01\x00"),
    BinarySignature("GZIP", b"\x1f\x8b"),
    # "BZh" plus a level digit, then the first block's own magic, 0x314159265359.
    BinarySignature("BZIP2", b"BZh", confirm=(4, b"1AY&SY")),
    BinarySignature("XZ", b"\xfd\x37\x7a\x58\x5a\x00"),
    # The tar magic is at byte 257, inside the first member's header block.
    BinarySignature("TAR", b"ustar", offset=257),
    # Executables / object files
    BinarySignature("ELF", b"\x7fELF"),
    BinarySignature("Mach-O 32", b"\xfe\xed\xfa\xce"),
    BinarySignature("Mach-O 64", b"\xfe\xed\xfa\xcf"),
    BinarySignature("Mach-O reverse", b"\xcf\xfa\xed\xfe"),
    # "MZ" alone is two letters. e_lfanew, at 0x3C, is a four byte offset to
    # the PE header that is always small, so its top two bytes are zero.
    BinarySignature("PE", b"MZ", confirm=(0x3E, b"\x00\x00")),
    # Databases
    BinarySignature("SQLite", b"SQLite format 3"),
    # Media
    #
    # Deliberately no "ID3": the tag header carries no fixed bytes to confirm
    # it with - the version, flags and syncsafe length are all variable - and
    # those length bytes are zero in every tag under 2 MB, so a tagged MP3 is
    # caught by the null byte rule below. A bare frame is caught here.
    BinarySignature("MP3 frame", b"\xff\xfb"),
    BinarySignature("FLAC", b"fLaC"),
)


def is_binary_content(file_path: str, sample_size: int = _MAX_SAMPLE_SIZE) -> bool:
    """
    Determine whether a file appears to contain binary content.

    Reads only a bounded prefix of the file (default 8 KB) to avoid loading
    large files entirely into memory. Detection is content-based and does not
    rely on file extensions.

    The prefix is classified in this order:

    1. A UTF-16 / UTF-32 byte order mark means text. The null bytes that make
       up half of a UTF-16 file would otherwise be read as binary evidence,
       which is how ordinary Windows-encoded source files ended up skipped.
    2. A known binary format signature (PNG, JPEG, PDF, ZIP, ELF, ...) means
       binary. Each signature carries the offset it lives at, and the ones made
       only of printable characters carry a second anchored run to confirm
       them, so a text file that happens to begin "BM" or "MZ" is not mistaken
       for one.
    3. A null byte outside a BOM-declared encoding means binary.
    4. Otherwise the sample is decoded through the shared fallback chain in
       :mod:`repo2readme.utils.text_encoding`. A multi-byte character cut in
       half by the sample boundary is tolerated, so a valid UTF-8 file is no
       longer called binary because of where byte 8192 lands. Because the last
       encoding in that chain cannot fail, the decoded text is then judged on
       its density of control characters: source files in a legacy single-byte
       encoding pass, binaries that happen to contain no null byte in their
       first 8 KB do not.

    Args:
        file_path: Absolute or relative path to the file to inspect.
        sample_size: Maximum number of bytes to read. Defaults to 8192.

    Returns:
        True if the file appears to be binary, False if it appears to be
        plain text (including UTF-8, UTF-16 and legacy single-byte encodings).

    Raises:
        OSError: If the file cannot be opened or read due to an I/O error.
        ValueError: If ``file_path`` is empty.
    """
    if not file_path:
        raise ValueError("file_path must not be empty")

    with open(file_path, "rb") as f:
        sample = f.read(sample_size)

    if not sample:
        return False

    # A byte order mark is an explicit declaration of a text encoding, so it is
    # checked before anything else - including the null-byte rule, which every
    # UTF-16 file would otherwise trip on its second byte.
    bom_encoding = encoding_from_bom(sample)

    # Check for known binary signatures. Matched against the whole sample
    # rather than a twelve byte prefix, because a signature does not have to
    # live at the start of the file.
    if any(signature.matches(sample) for signature in BINARY_SIGNATURES):
        return True

    if bom_encoding is None and b"\x00" in sample:
        # Null bytes are a strong, extension-independent indicator of binary
        # data in any encoding that does not declare itself.
        return True

    decoded = decode_bytes(sample, allow_truncated=True)
    if decoded is None:
        return True

    return not looks_like_text(decoded.text)
