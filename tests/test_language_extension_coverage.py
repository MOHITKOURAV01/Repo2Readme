"""
Tests for the language extension table (issue #177).

The table covered 22 languages and stopped, and the content fallback answered
anyway: a `.cc` file was JSON, a `.mjs` file was Python, a `.ex` file was Ruby.
The language is stated to the model above the source and forms part of the
summary cache key, so a confident wrong answer is worse than `unknown`.

Covers:
- the extensions the issue listed, and the families around them
- the content fallback no longer naming JSON from two punctuation characters
- one table, shared with the dependency graph, so the two cannot disagree
- the detection order and the dotfile lookup are unchanged
"""

from __future__ import annotations

import pytest

from repo2readme.dependency_graph import _detect_language_from_ext
from repo2readme.utils.detect_language import (
    EXTENSION_LANGUAGE_MAP,
    PARSEABLE_LANGUAGES,
    _detect_by_content,
    detect_lang,
    language_for_extension,
)

# The cases from the issue, with the answer each one used to give.
REGRESSIONS = [
    pytest.param("src/server.mjs", "import x from './x.js';\n", "javascript", id="mjs"),
    pytest.param("src/legacy.cjs", "module.exports = 1;\n", "javascript", id="cjs"),
    pytest.param("include/engine.h", "#ifndef E_H\nvoid run(void);\n", "c", id="h"),
    pytest.param("src/engine.cc", '#include "engine.h"\n', "cpp", id="cc"),
    pytest.param("src/App.vue", "<template><div/></template>\n", "vue", id="vue"),
    pytest.param("src/App.svelte", "<script>let x=1;</script>\n", "svelte", id="svelte"),
    pytest.param("api/types.pyi", "def f(x: int) -> str: ...\n", "python", id="pyi"),
    pytest.param("infra/main.tf", 'resource "b" "c" {}\n', "terraform", id="tf"),
    pytest.param("api/schema.proto", 'syntax = "proto3";\n', "protobuf", id="proto"),
    pytest.param("lib/core.ex", "defmodule Core do\nend\n", "elixir", id="ex"),
    pytest.param("src/main.dart", "void main() {}\n", "dart", id="dart"),
]


@pytest.mark.parametrize("path,content,expected", REGRESSIONS)
def test_the_extensions_from_the_issue(path, content, expected):
    assert detect_lang(path, content) == expected


FAMILIES = [
    # C family
    (".hpp", "cpp"), (".hh", "cpp"), (".cxx", "cpp"), (".m", "objective-c"),
    (".mm", "objective-cpp"),
    # JavaScript / TypeScript
    (".mts", "typescript"), (".cts", "typescript"),
    # Interface and schema definitions
    (".graphql", "graphql"), (".gql", "graphql"), (".thrift", "thrift"),
    (".prisma", "prisma"),
    # Infrastructure
    (".hcl", "hcl"), (".tfvars", "terraform"), (".nix", "nix"),
    # Functional
    (".exs", "elixir"), (".erl", "erlang"), (".hs", "haskell"), (".elm", "elm"),
    (".ml", "ocaml"), (".clj", "clojure"),
    # Scripting
    (".lua", "lua"), (".jl", "julia"), (".cr", "crystal"), (".pm", "perl"),
    # Documentation
    (".mdx", "mdx"), (".adoc", "asciidoc"), (".tex", "tex"),
    # Web
    (".sass", "sass"), (".hbs", "handlebars"), (".htm", "html"),
]


@pytest.mark.parametrize("extension,expected", FAMILIES)
def test_the_families_around_them(extension, expected):
    assert language_for_extension(extension) == expected


def test_an_unknown_extension_is_still_empty():
    assert language_for_extension(".xyz") == ""
    assert detect_lang("notes.xyz") == "unknown"


def test_the_lookup_is_case_insensitive():
    assert language_for_extension(".CC") == "cpp"
    assert language_for_extension(".TF") == "terraform"


# ---------------------------------------------------------------------------
# The content fallback
# ---------------------------------------------------------------------------


class TestContentFallback:
    @pytest.mark.parametrize(
        "content",
        [
            '#include "engine.h"\nvoid run(void) {}\n',
            'resource "aws_s3_bucket" "b" {}\n',
            'syntax = "proto3";\nmessage M {}\n',
            "void main() { print('x'); }\n",
        ],
        ids=["cpp", "terraform", "protobuf", "dart"],
    )
    def test_a_brace_and_a_quote_no_longer_mean_json(self, content):
        assert _detect_by_content(content) != "json"

    @pytest.mark.parametrize(
        "content",
        ['{"name": "test"}', '{"a":1}', '[\n  {"id": 1},\n  {"id": 2}\n]'],
        ids=["object", "compact", "array"],
    )
    def test_real_json_is_still_recognised(self, content):
        assert _detect_by_content(content) == "json"

    def test_a_groovy_pipeline_is_groovy(self):
        content = "pipeline {\n  agent any\n  stages {\n    stage('b') { sh 'x' }\n  }\n}\n"

        assert _detect_by_content(content) == "groovy"

    def test_the_other_rules_are_unchanged(self):
        assert _detect_by_content("import os\nclass A:\n    pass\n") == "python"
        assert _detect_by_content("FROM python:3.12\nRUN pip install x\n") == "dockerfile"
        assert _detect_by_content("interface U {\n  readonly id: number;\n}\n") == "typescript"

    def test_empty_content_has_no_answer(self):
        assert _detect_by_content("") is None


# ---------------------------------------------------------------------------
# One table
# ---------------------------------------------------------------------------


class TestSharedTable:
    @pytest.mark.parametrize(
        "extension", [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]
    )
    def test_the_dependency_graph_agrees_with_the_detector(self, extension):
        assert _detect_language_from_ext(extension) == language_for_extension(extension)

    def test_mjs_was_the_disagreement(self):
        # The dependency graph resolved .mjs imports as JavaScript while
        # detect_lang called the same file Python.
        assert detect_lang("src/server.mjs", "import x from './x.js';\n") == "javascript"
        assert _detect_language_from_ext(".mjs") == "javascript"

    def test_a_language_with_no_import_parser_is_not_claimed(self):
        # The detector knows what these are; the graph has nothing to parse
        # them with and says so.
        for extension in (".vue", ".tf", ".proto", ".cc", ".ex"):
            assert language_for_extension(extension) != ""
            assert _detect_language_from_ext(extension) == ""

    def test_every_parseable_language_is_reachable(self):
        reachable = {
            language
            for language in EXTENSION_LANGUAGE_MAP.values()
            if language in PARSEABLE_LANGUAGES
        }

        assert reachable == set(PARSEABLE_LANGUAGES)

    def test_no_extension_is_missing_its_leading_dot(self):
        missing = [key for key in EXTENSION_LANGUAGE_MAP if not key.startswith(".")]

        assert missing == []

    def test_keys_are_lowercase(self):
        assert [key for key in EXTENSION_LANGUAGE_MAP if key != key.lower()] == []


# ---------------------------------------------------------------------------
# Nothing else about detection changed
# ---------------------------------------------------------------------------


class TestOrderIsUnchanged:
    def test_the_extension_still_wins_over_the_content(self):
        assert detect_lang("app.py", "FROM python:3.12\nRUN pip install x\n") == "python"

    def test_the_filename_map_still_runs_before_the_shebang(self):
        assert detect_lang("Gemfile", "#!/usr/bin/env python\n") == "ruby"

    def test_the_shebang_still_runs_before_the_content(self):
        assert detect_lang("script", "#!/bin/bash\nimport os\nclass A:\n") == "bash"

    def test_an_ignore_file_is_a_list_of_globs_not_a_dockerfile(self):
        assert detect_lang("/repo/.dockerignore") == "gitignore"
        assert detect_lang("/repo/.npmignore") == "gitignore"

    def test_an_unknown_dotfile_is_still_unknown(self):
        assert detect_lang("/repo/.mysteryrc") == "unknown"

    def test_a_known_dotfile_resolves_from_any_path_shape(self):
        assert (
            detect_lang(".editorconfig")
            == detect_lang("app/.editorconfig")
            == detect_lang("/tmp/clone/app/.editorconfig")
            == "ini"
        )
