"""
Tests for carrying the detected language through the pipeline (issue #128).

Covers:
- create_document keeps language, file_size and mtime
- the summarizer uses the language the traversal detected
- the cases that were wrong before: Gemfile, Jenkinsfile, extensionless scripts
- the fallback for hand-built documents, and what it is given to work with
- SummaryCache entries carry a real mtime
- dotfile extension detection works for a path, not only a bare filename
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from repo2readme.cache import SummaryCache
from repo2readme.loaders.traversal.pipeline import TraversalPipeline
from repo2readme.loaders.traversal.stages import (
    FilteredFile,
    create_document,
    detect_file_language,
    extract_file_metadata,
)
from repo2readme.services.summarization import (
    generate_all_summaries,
    resolve_language,
)
from repo2readme.utils.detect_language import detect_lang

# Files whose language the extension alone cannot decide. The second element is
# what the pipeline detects; before this change the summarizer saw the third.
AMBIGUOUS_FILES = [
    ("Gemfile", "source 'https://rubygems.org'\ngem 'rails'\ngem 'puma'\n", "ruby"),
    (
        "Jenkinsfile",
        "pipeline {\n  agent any\n  stages {\n    stage('b') { sh 'make' }\n  }\n}\n",
        "groovy",
    ),
    ("Rakefile", "task :default do\n  puts 'hi'\nend\n", "ruby"),
    ("Vagrantfile", "Vagrant.configure('2') do |config|\nend\n", "ruby"),
    ("deploy", "#!/usr/bin/env bash\nset -e\nkubectl apply -f k8s/\n", "bash"),
    ("entrypoint", "#!/usr/bin/env python3\nimport sys\nprint(sys.argv)\n", "python"),
]


def _document_for(tmp_path, name, body):
    """Run one file through the metadata / language / document stages."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")

    filtered = FilteredFile(
        absolute_path=str(path), relative_path=name, file_name=name
    )
    metadata = extract_file_metadata(filtered, body)
    language = detect_file_language(metadata, body)
    metadata = type(metadata)(
        absolute_path=metadata.absolute_path,
        relative_path=metadata.relative_path,
        file_name=metadata.file_name,
        file_type=metadata.file_type,
        file_size=metadata.file_size,
        mtime=metadata.mtime,
        language=language,
    )
    return create_document(metadata, body)


# ---------------------------------------------------------------------------
# create_document
# ---------------------------------------------------------------------------


class TestCreateDocumentMetadata:
    def test_language_is_present(self, tmp_path):
        doc = _document_for(tmp_path, "app.py", "import os\nprint(os)\n")
        assert doc.metadata["language"] == "python"

    def test_existing_keys_are_unchanged(self, tmp_path):
        doc = _document_for(tmp_path, "app.py", "x = 1\n")
        assert doc.metadata["file_name"] == "app.py"
        assert doc.metadata["file_type"] == ".py"
        assert doc.metadata["relative_path"] == "app.py"
        assert doc.metadata["file_path"].endswith("/app.py")

    def test_file_size_is_present(self, tmp_path):
        body = "x = 1\n"
        doc = _document_for(tmp_path, "app.py", body)
        assert doc.metadata["file_size"] == len(body)

    def test_mtime_is_a_real_timestamp(self, tmp_path):
        doc = _document_for(tmp_path, "app.py", "x = 1\n")
        assert doc.metadata["mtime"] > 0
        assert doc.metadata["mtime"] <= time.time() + 1

    def test_metadata_key_order_is_stable(self, tmp_path):
        doc = _document_for(tmp_path, "app.py", "x = 1\n")
        assert list(doc.metadata) == [
            "file_path",
            "file_name",
            "file_type",
            "relative_path",
            "language",
            "file_size",
            "mtime",
        ]

    def test_unreadable_file_falls_back_without_raising(self, tmp_path):
        filtered = FilteredFile(
            absolute_path=str(tmp_path / "gone.py"),
            relative_path="gone.py",
            file_name="gone.py",
        )
        metadata = extract_file_metadata(filtered, "x = 1\n")
        assert metadata.file_size == len("x = 1\n")
        assert metadata.mtime == 0.0


# ---------------------------------------------------------------------------
# The pipeline end to end
# ---------------------------------------------------------------------------


class TestPipelineCarriesLanguage:
    @pytest.mark.parametrize("name,body,expected", AMBIGUOUS_FILES)
    def test_ambiguous_files_reach_the_summarizer_correctly(
        self, tmp_path, name, body, expected
    ):
        (tmp_path / name).write_text(body, encoding="utf-8")
        documents, _ = TraversalPipeline(folder_path=str(tmp_path)).run()

        assert len(documents) == 1
        metadata = dict(documents[0].metadata)
        assert metadata["language"] == expected
        assert resolve_language(metadata, body) == expected

    def test_jenkinsfile_is_no_longer_summarized_as_json(self, tmp_path):
        """The extension-only path matched the JSON marker set on `{`, `"`, `}`.

        It no longer does - the JSON rule names JSON's punctuation pairs now,
        not the individual characters (issue #177) - so the extension-only path
        falls through to the Groovy rule instead of claiming JSON. The point of
        this test is unchanged: what the pipeline carries is the right answer,
        arrived at from the whole path rather than the bare extension.
        """
        body = "pipeline {\n  agent any\n  stages {\n    stage('b') { sh 'make' }\n  }\n}\n"
        (tmp_path / "Jenkinsfile").write_text(body, encoding="utf-8")
        documents, _ = TraversalPipeline(folder_path=str(tmp_path)).run()

        metadata = dict(documents[0].metadata)
        assert detect_lang(metadata["file_type"], body) != "json"
        assert resolve_language(metadata, body) == "groovy"

    def test_every_document_has_a_language(self, tmp_path):
        for name, body, _ in AMBIGUOUS_FILES:
            (tmp_path / name).write_text(body, encoding="utf-8")
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

        documents, _ = TraversalPipeline(folder_path=str(tmp_path)).run()
        assert documents
        for document in documents:
            assert document.metadata["language"]
            assert document.metadata["mtime"] > 0


# ---------------------------------------------------------------------------
# resolve_language
# ---------------------------------------------------------------------------


class TestResolveLanguage:
    def test_prefers_the_metadata_value(self):
        metadata = {"language": "groovy", "file_type": ".json"}
        assert resolve_language(metadata, '{"a": 1}') == "groovy"

    def test_falls_back_when_the_key_is_missing(self):
        metadata = {"file_path": "/repo/app.py", "file_type": ".py"}
        assert resolve_language(metadata, "x = 1\n") == "python"

    def test_falls_back_when_the_language_is_unknown(self):
        metadata = {"language": "unknown", "relative_path": "app.py"}
        assert resolve_language(metadata, "x = 1\n") == "python"

    def test_fallback_uses_a_path_not_the_bare_extension(self):
        """The old call passed file_type, so filename rules never fired."""
        metadata = {"relative_path": "ops/Gemfile", "file_type": ""}
        assert resolve_language(metadata, "gem 'rails'\n") == "ruby"

    def test_fallback_prefers_relative_path_over_absolute(self):
        metadata = {
            "relative_path": "ops/Rakefile",
            "file_path": "/tmp/clone/ops/Rakefile",
        }
        assert resolve_language(metadata, "task :default\n") == "ruby"

    def test_empty_metadata_does_not_raise(self):
        assert resolve_language({}, "") == "unknown"


# ---------------------------------------------------------------------------
# What reaches the summarizer and the cache
# ---------------------------------------------------------------------------


class TestSummarizationUsesTheDetectedLanguage:
    def _documents(self, tmp_path):
        (tmp_path / "Gemfile").write_text("gem 'rails'\n", encoding="utf-8")
        documents, _ = TraversalPipeline(folder_path=str(tmp_path)).run()
        return [
            {"content": d.page_content, "metadata": dict(d.metadata)}
            for d in documents
        ]

    def test_language_passed_to_summarize_file(self, tmp_path):
        documents = self._documents(tmp_path)
        cache = SummaryCache(
            cache_dir=str(tmp_path / "cache"),
            config={},
            prompt_template_hash="h",
            autosave=False,
        )

        seen = {}

        def fake_summarize(file_path, language, content, **kwargs):
            seen["language"] = language
            return {"file_path": file_path, "description": "ok"}

        with patch(
            "repo2readme.services.summarization.summarize_file",
            side_effect=fake_summarize,
        ):
            summaries, errors = generate_all_summaries(
                documents=documents, summary_cache=cache
            )

        assert errors == []
        assert len(summaries) == 1
        assert seen["language"] == "ruby"

    def test_cache_entry_records_a_real_mtime(self, tmp_path):
        documents = self._documents(tmp_path)
        cache = SummaryCache(
            cache_dir=str(tmp_path / "cache"),
            config={},
            prompt_template_hash="h",
            autosave=False,
        )

        with patch(
            "repo2readme.services.summarization.summarize_file",
            side_effect=lambda file_path, language, content, **kw: {
                "file_path": file_path,
                "description": "ok",
            },
        ):
            generate_all_summaries(documents=documents, summary_cache=cache)

        cache.flush()
        entries = cache.get_deleted_files(set())
        assert len(entries) == 1
        assert entries[0]["mtime"] > 0
        assert entries[0]["language"] == "ruby"

    def test_cache_round_trip_hits_on_the_second_run(self, tmp_path):
        """A stable language is what makes the cache key stable."""
        documents = self._documents(tmp_path)
        cache = SummaryCache(
            cache_dir=str(tmp_path / "cache"),
            config={},
            prompt_template_hash="h",
            autosave=False,
        )

        calls = {"n": 0}

        def fake_summarize(file_path, language, content, **kwargs):
            calls["n"] += 1
            return {"file_path": file_path, "description": "ok"}

        with patch(
            "repo2readme.services.summarization.summarize_file",
            side_effect=fake_summarize,
        ):
            generate_all_summaries(documents=documents, summary_cache=cache)
            generate_all_summaries(documents=documents, summary_cache=cache)

        assert calls["n"] == 1

    def test_hand_built_documents_still_work(self, tmp_path):
        """A caller that never went through the pipeline has no language key."""
        documents = [
            {
                "content": "x = 1\n",
                "metadata": {"file_path": "/repo/app.py", "file_type": ".py"},
            }
        ]
        cache = SummaryCache(
            cache_dir=str(tmp_path / "cache"),
            config={},
            prompt_template_hash="h",
            autosave=False,
        )

        seen = {}

        def fake_summarize(file_path, language, content, **kwargs):
            seen["language"] = language
            return {"file_path": file_path, "description": "ok"}

        with patch(
            "repo2readme.services.summarization.summarize_file",
            side_effect=fake_summarize,
        ):
            summaries, errors = generate_all_summaries(
                documents=documents, summary_cache=cache
            )

        assert errors == []
        assert len(summaries) == 1
        assert seen["language"] == "python"


# ---------------------------------------------------------------------------
# Dotfile extensions
# ---------------------------------------------------------------------------


class TestDotfileExtensionDetection:
    # ".dockerignore" is a list of globs, not Dockerfile syntax, and maps to
    # the ignore-file language now (issue #177). What is under test here is the
    # dotfile lookup itself, which is unchanged.
    def test_bare_filename(self):
        assert detect_lang(".dockerignore") == "gitignore"

    def test_relative_path(self):
        assert detect_lang("app/.dockerignore") == "gitignore"

    def test_absolute_path(self):
        """The rule compared the whole path against ".", so this used to miss."""
        assert detect_lang("/tmp/clone/app/.dockerignore") == "gitignore"

    def test_windows_style_path(self):
        assert detect_lang(".dockerignore") == detect_lang("app/.dockerignore")

    def test_ordinary_extensions_are_unaffected(self):
        assert detect_lang("/repo/src/app.py") == "python"
        assert detect_lang("/repo/src/app.ts") == "typescript"

    def test_unknown_dotfile_is_still_unknown(self):
        assert detect_lang("/repo/.mysteryrc") == "unknown"
