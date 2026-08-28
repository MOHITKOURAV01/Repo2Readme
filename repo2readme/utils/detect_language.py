"""
Language detection module for repo2readme.

Detection order:
1. Extension-based detection (primary, highest priority)
2. Filename-based detection (for extensionless files)
3. Unix shebang detection (first line of file)
4. Lightweight content-based detection (rule-based heuristics)
5. Unknown (fallback)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# 1. EXTENSION → LANGUAGE MAP (existing, unchanged)
# ---------------------------------------------------------------------------
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    # --- Python
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".pyx": "cython",
    ".pxd": "cython",
    # --- JavaScript / TypeScript
    #
    # .mjs and .cjs were missing here while dependency_graph resolved imports
    # in both, so the two tables disagreed about the same file. They are shared
    # now: see language_for_extension.
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # --- Component frameworks
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    # --- C family
    #
    # A .h can be C, C++ or Objective-C and nothing in the name says which.
    # Linguist calls it C, which is the safe answer: the summarizer is being
    # told what kind of file it is looking at, and "c" is right about the
    # syntax either way.
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".ipp": "cpp",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".cs": "csharp",
    ".csx": "csharp",
    # --- JVM
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".edn": "clojure",
    # --- Systems
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
    ".nim": "nim",
    ".d": "d",
    ".v": "v",
    ".swift": "swift",
    # --- Functional
    ".hs": "haskell",
    ".lhs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".elm": "elm",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".purs": "purescript",
    ".scm": "scheme",
    ".rkt": "racket",
    ".lisp": "lisp",
    ".el": "lisp",
    # --- Scripting
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".tcl": "tcl",
    ".r": "r",
    ".jl": "julia",
    ".dart": "dart",
    ".cr": "crystal",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "bash",
    ".ksh": "bash",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".awk": "awk",
    ".vb": "vbnet",
    ".pas": "pascal",
    ".f90": "fortran",
    ".f95": "fortran",
    ".sol": "solidity",
    # --- Interface and schema definitions, which a README is usually about
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".thrift": "thrift",
    ".capnp": "capnproto",
    ".avsc": "avro",
    ".prisma": "prisma",
    ".sql": "sql",
    # --- Infrastructure
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "hcl",
    ".bicep": "bicep",
    ".nix": "nix",
    # --- Markdown / documentation
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "mdx",
    ".rst": "rst",
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".org": "org",
    ".tex": "tex",
    ".bib": "bibtex",
    # --- Data / config
    ".json": "json",
    ".jsonc": "json",
    ".json5": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".properties": "ini",
    ".xml": "xml",
    ".xsd": "xml",
    ".plist": "xml",
    ".csv": "csv",
    ".tsv": "tsv",
    # --- Web
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".styl": "stylus",
    ".pug": "pug",
    ".hbs": "handlebars",
    ".ejs": "ejs",
    ".twig": "twig",
    ".liquid": "liquid",
    ".njk": "nunjucks",
    # --- Dotfiles. splitext gives a leading-dot basename no extension, so the
    # whole name is the lookup key; see _detect_by_extension.
    ".bashrc": "bash",
    ".zshrc": "bash",
    ".bash_profile": "bash",
    ".profile": "bash",
    ".envrc": "bash",
    ".editorconfig": "ini",
    ".gitmodules": "ini",
    ".babelrc": "json",
    ".eslintrc": "json",
    ".prettierrc": "json",
    ".stylelintrc": "json",
    # An ignore file is a list of globs. ".dockerignore" used to be mapped to
    # "dockerfile", which it shares no syntax with; Linguist groups the whole
    # family as one thing instead.
    ".gitignore": "gitignore",
    ".dockerignore": "gitignore",
    ".npmignore": "gitignore",
    ".eslintignore": "gitignore",
    ".prettierignore": "gitignore",
}


# Languages the dependency graph knows how to parse imports for. Kept here so
# there is one table, rather than one per module that needs to ask.
PARSEABLE_LANGUAGES = frozenset({"python", "javascript", "typescript"})


def language_for_extension(extension: str) -> str:
    """Language for a file extension, or ``""`` when it is not in the table.

    The lookup other modules should use, so a new extension is added in one
    place. ``dependency_graph`` carried its own copy that had ``.mjs`` and
    ``.cjs`` in it while this one did not, and the two disagreed about the same
    file.
    """
    return EXTENSION_LANGUAGE_MAP.get(extension.lower(), "")


# ---------------------------------------------------------------------------
# 2. FILENAME → LANGUAGE MAP (for common extensionless files)
# ---------------------------------------------------------------------------
# Note: cargo.toml, docker-compose.yml/yaml are omitted because extension
# detection runs first, so they are unreachable.
FILENAME_LANGUAGE_MAP: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "jenkinsfile": "groovy",
    "procfile": "procfile",
    "cmakelists.txt": "cmake",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "brewfile": "ruby",
    "vagrantfile": "ruby",
    # Additional common filenames
    "justfile": "just",
    "snakefile": "python",
    "gemfile.lock": "ruby",
    "guardfile": "ruby",
    "capfile": "ruby",
    "podfile": "ruby",
    "fastfile": "ruby",
    "appfile": "ruby",
    "matchfile": "ruby",
    "pluginfile": "ruby",
    "berksfile": "ruby",
    "thorfile": "ruby",
    "dangerfile": "ruby",
    "gnumakefile": "makefile",
    "makefile.am": "makefile",
    "makefile.in": "makefile",
    "meson.build": "meson",
    "build": "bazel",
    "workspace": "bazel",
}

# ---------------------------------------------------------------------------
# 3. SHEBANG PATTERNS
# ---------------------------------------------------------------------------
# Ordered from most-specific to least-specific to avoid false positives.
SHEBANG_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Python (direct path: /usr/bin/python* or env path: env python*)
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/))python(\d+(?:\.\d+)?)?(?:\s|$)"
        ),
        "python",
    ),
    (re.compile(r"^\s*#!\s*/bin/python(\d+(?:\.\d+)?)?(?:\s|$)"), "python"),
    # Bash
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))bash(?:\s|$)"
        ),
        "bash",
    ),
    # Sh
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))sh(?:\s|$)"
        ),
        "sh",
    ),
    # Node
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))node(?:\s|$)"
        ),
        "javascript",
    ),
    # Deno / Bun
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/))(?:deno|bun)(?:\s|$)"
        ),
        "javascript",
    ),
    # Ruby
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))ruby(?:\s|$)"
        ),
        "ruby",
    ),
    # Perl
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))perl(?:\s|$)"
        ),
        "perl",
    ),
    # PHP
    (
        re.compile(
            r"^\s*#!\s*(?:(?:/usr/bin/env\s+)|(?:/usr/bin/)|(?:/bin/))php(?:\s|$)"
        ),
        "php",
    ),
    # Racket
    (re.compile(r"^\s*#!\s*/usr/bin/env\s+racket(?:\s|$)"), "racket"),
    # Lua
    (re.compile(r"^\s*#!\s*/usr/bin/env\s+lua(?:\s|$)"), "lua"),
    # Awk
    (re.compile(r"^\s*#!\s*/usr/bin/env\s+awk(?:\s|$)"), "awk"),
    # Guile (Scheme)
    (re.compile(r"^\s*#!\s*/usr/bin/env\s+guile(?:\s|$)"), "scheme"),
    # Tcl
    (re.compile(r"^\s*#!\s*/usr/bin/env\s+tclsh(?:\s|$)"), "tcl"),
]

# ---------------------------------------------------------------------------
# 4. CONTENT-BASED DETECTION RULES
# ---------------------------------------------------------------------------
# Each entry is a (language, list-of-markers) pair.
# Markers are checked against the first 8 KB of content.
# More specific markers should come first for each language.
# Shared keywords (e.g. `def`, `class`) are omitted to avoid false positives
# between similar-looking languages. Instead we use more distinctive tokens.
@dataclass(frozen=True)
class ContentRule:
    """Markers that suggest a language, and how many have to be present.

    The threshold is per rule because the rules are not equally strong. Two
    markers is enough for Python's ``if __name__ ==`` and ``self.``; it was far
    too little for the JSON rule, which listed ``{``, ``[``, ``"``, ``: `` and
    ``}`` and so matched almost any source file in any language.
    """

    language: str
    markers: tuple[str, ...]
    minimum: int = 2

    def score(self, sample: str) -> int:
        return sum(1 for marker in self.markers if marker in sample)

    def matches(self, sample: str) -> bool:
        return self.score(sample) >= self.minimum


CONTENT_RULES: tuple[ContentRule, ...] = (
    # Dockerfile
    ContentRule("dockerfile", ("FROM ", "RUN ", "COPY ", "CMD ", "ENTRYPOINT ")),
    # Python
    ContentRule(
        "python",
        ("if __name__ ==", "import ", "from ", "self.", "def ", "class "),
    ),
    # TypeScript
    ContentRule("typescript", ("interface ", "implements ", "readonly ", "type ")),
    # JavaScript (less specific than TypeScript, so listed after)
    ContentRule(
        "javascript",
        ("module.exports", "require(", "=>", "const ", "let ", "function "),
    ),
    # Shell
    ContentRule(
        "bash",
        ("#!/", "echo ", "export ", " fi\n", "\nfi\n", "then ", "done ", "else "),
    ),
    # YAML
    ContentRule("yaml", ("---\n", ":\n  ", "  - ", ": ")),
    # JSON. The markers name JSON's own punctuation pairs rather than the
    # individual characters: a brace and a quote appear in every C-like
    # language, but `{"` and `": ` do not.
    ContentRule("json", ('{"', '["', '": ', '":', "},", "],", ', "')),
    # Markdown
    ContentRule("markdown", ("# ", "## ", "```", "---")),
    # Groovy / Jenkinsfile
    ContentRule("groovy", ("pipeline {", "stages {", "stage(", "agent ")),
    # Makefile
    ContentRule("makefile", ("CC=", "CFLAGS=", "LDFLAGS=", "$@", "$<", ":=", "PHONY")),
    # Ruby
    ContentRule("ruby", ("require ", "gem ", "puts ", "end\n", "module ", "class ")),
    # Perl
    ContentRule("perl", ("use strict", "use warnings", "my $", 'print "')),
    # PHP
    ContentRule("php", ("<?php", "function ", "echo ", "$this->")),
)

# Maximum bytes to read for content-based detection
_MAX_CONTENT_BYTES = 8192


# ===================================================================
# Helper functions
# ===================================================================


def _detect_by_extension(path: str) -> Optional[str]:
    """
    Detect language based on file extension.
    Returns language string or None if no match.
    """
    basename = os.path.basename(path)
    _, extension = os.path.splitext(basename)

    # Handle dotfiles (e.g. ".dockerignore"): splitext treats a leading-dot
    # basename as having no extension, so the whole name is the lookup key.
    # This has to be the basename rather than ``path``, or the rule only fires
    # for a bare filename and never for the absolute paths the pipeline passes.
    if not extension and basename.startswith("."):
        extension = basename

    return language_for_extension(extension) or None


def _detect_by_filename(path: str) -> Optional[str]:
    """
    Detect language based on filename for common extensionless files.
    Extracts just the basename for matching.
    Returns language string or None if no match.
    """
    filename = os.path.basename(path)
    return FILENAME_LANGUAGE_MAP.get(filename.lower(), None)


def _detect_by_shebang(content: str) -> Optional[str]:
    """
    Detect language by parsing the first line of content for a Unix shebang.
    Ignores leading whitespace before the shebang marker.
    Returns language string or None if no match.
    """
    if not content:
        return None

    # Extract the first line (up to first newline)
    first_line = content.split("\n")[0].strip()

    # Only check lines that look like shebangs (start with #!)
    if not first_line.startswith("#!"):
        return None

    # Try each shebang pattern
    for pattern, language in SHEBANG_PATTERNS:
        if pattern.match(first_line):
            return language

    return None


def _detect_by_content(content: str) -> Optional[str]:
    """
    Lightweight rule-based content detection.
    Inspects only the first 8 KB of content.
    Uses simple string matching (no heavy parsing libraries).
    Returns language string or None if no match.
    """
    if not content:
        return None

    # Limit content to avoid scanning large files
    sample = content[:_MAX_CONTENT_BYTES]

    for rule in CONTENT_RULES:
        if rule.matches(sample):
            return rule.language

    return None


def _read_file_content(file_path: str) -> Optional[str]:
    """
    Safely read the first portion of a file for detection purposes.
    Handles binary files, permission errors, and other I/O issues.
    Returns content string or None if the file cannot be read.
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(_MAX_CONTENT_BYTES)

        # Check for null bytes indicating binary content
        if b"\0" in raw:
            return None

        # Decode with UTF-8 (ignore invalid bytes first; only fall back to latin-1 if needed)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="ignore")

    except (IOError, OSError, PermissionError):
        return None


# ===================================================================
# Main API
# ===================================================================


def detect_lang(path: str, content: Optional[str] = None) -> str:
    """
    Detect the programming or markup language of a file.

    Detection order:
    1. Extension-based detection (highest priority, backward compatible)
    2. Filename-based detection (for extensionless files)
    3. Unix shebang detection (first line)
    4. Lightweight content-based detection
    5. Unknown (fallback)

    Args:
        path: File path or filename (can be just an extension like ".py").
        content: Optional file content as string. If not provided and the path
                 points to an existing file, it will be read automatically.

    Returns:
        A string identifying the language (e.g., "python", "javascript", "unknown").
    """
    # Step 1: Extension-based detection (highest priority)
    result = _detect_by_extension(path)
    if result:
        return result

    # Step 2: Filename-based detection
    result = _detect_by_filename(path)
    if result:
        return result

    # Obtain content for shebang and content-based detection
    file_content: Optional[str] = content

    # If content was not passed, try to read the file
    if file_content is None:
        # If path looks like a valid file path, try to read it
        if os.path.isfile(path):
            file_content = _read_file_content(path)

    if file_content is None:
        return "unknown"

    # Step 3: Shebang detection
    result = _detect_by_shebang(file_content)
    if result:
        return result

    # Step 4: Content-based detection (final fallback)
    result = _detect_by_content(file_content)
    if result:
        return result

    # Step 5: Unknown
    return "unknown"