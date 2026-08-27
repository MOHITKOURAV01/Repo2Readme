"""One file on disk, one document.

A symbolic link pointing inside the repository is followed, so the file it names
is reached twice - once as itself, once through the link. Both copies used to be
loaded, summarized, cached and billed. Directory links have always been
deduplicated; files were the exception.
"""

import os

import pytest

from repo2readme.dependency_graph import build_dependency_graph
from repo2readme.loaders.traversal import TraversalPipeline
from repo2readme.loaders.traversal.stages import discover_files


def _skip_if_no_symlink_support(tmp_path):
    probe = tmp_path / "probe"
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    try:
        probe.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    probe.unlink()
    target.unlink()


@pytest.fixture
def repo(tmp_path):
    _skip_if_no_symlink_support(tmp_path)
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "app.py").write_text("print('hello')\n" * 5, encoding="utf-8")
    return root


def analyzed(root, **kwargs):
    documents, context = TraversalPipeline(folder_path=str(root), **kwargs).run()
    return (
        sorted(doc.metadata["relative_path"] for doc in documents),
        context,
    )


class TestOneDocumentPerFile:
    def test_a_linked_file_is_analyzed_once(self, repo):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")

        paths, _ = analyzed(repo)

        assert paths == ["src/app.py"]

    def test_several_links_to_one_file_collapse(self, repo):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")
        (repo / "link.py").symlink_to(repo / "src" / "app.py")

        paths, _ = analyzed(repo)

        assert paths == ["src/app.py"]

    def test_the_duplicates_are_reported_against_the_path_that_was_kept(self, repo):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")
        (repo / "link.py").symlink_to(repo / "src" / "app.py")

        _, context = analyzed(repo)

        duplicates = sorted(
            entry for entry in context.skipped if "duplicate" in entry[1]
        )
        assert duplicates == [
            ("docs/app.py", "duplicate of src/app.py"),
            ("link.py", "duplicate of src/app.py"),
        ]

    def test_the_file_is_counted_once(self, repo):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")

        _, context = analyzed(repo)

        assert context.repository_metadata.file_count == 1
        assert context.repository_metadata.total_size == os.path.getsize(
            repo / "src" / "app.py"
        )

    def test_distinct_files_with_equal_contents_are_both_kept(self, repo):
        # Two real files that happen to match are two files. Only one path on
        # disk is a duplicate.
        (repo / "docs" / "copy.py").write_text(
            "print('hello')\n" * 5, encoding="utf-8"
        )

        paths, _ = analyzed(repo)

        assert paths == ["docs/copy.py", "src/app.py"]


class TestWhichPathIsKept:
    def test_the_real_file_beats_the_link(self, repo):
        # The link sorts first alphabetically; the real file still wins.
        (repo / "aaa.py").symlink_to(repo / "src" / "app.py")

        paths, _ = analyzed(repo)

        assert paths == ["src/app.py"]

    def test_between_two_links_the_shallower_path_wins(self, repo):
        # The real file lives in a directory the default rules ignore, so only
        # the two links compete and the tie-break is what decides.
        (repo / "dist").mkdir()
        (repo / "dist" / "generated.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "top.py").symlink_to(repo / "dist" / "generated.py")
        (repo / "docs" / "nested.py").symlink_to(repo / "dist" / "generated.py")

        paths, context = analyzed(repo)

        assert paths == ["src/app.py", "top.py"]
        assert ("docs/nested.py", "duplicate of top.py") in context.skipped

    def test_between_two_links_at_one_depth_the_earlier_name_wins(self, repo):
        (repo / "dist").mkdir()
        (repo / "dist" / "generated.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "zeta.py").symlink_to(repo / "dist" / "generated.py")
        (repo / "alpha.py").symlink_to(repo / "dist" / "generated.py")

        paths, context = analyzed(repo)

        assert paths == ["alpha.py", "src/app.py"]
        assert ("zeta.py", "duplicate of alpha.py") in context.skipped

    def test_the_choice_does_not_depend_on_walk_order(self, repo, monkeypatch):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")

        import repo2readme.loaders.traversal.stages as stages

        real_walk = os.walk

        def reversed_walk(top, *args, **kwargs):
            for current, dirs, files in real_walk(top, *args, **kwargs):
                dirs.sort(reverse=True)
                yield current, dirs, sorted(files, reverse=True)

        forward, _ = analyzed(repo)
        monkeypatch.setattr(stages.os, "walk", reversed_walk)
        backward, _ = analyzed(repo)

        assert forward == backward == ["src/app.py"]


class TestExistingSymlinkRulesAreUnchanged:
    def test_a_broken_link_is_still_reported_as_broken(self, repo):
        (repo / "dangling.py").symlink_to(repo / "src" / "missing.py")

        _, context = analyzed(repo)

        assert ("dangling.py", "broken symbolic link") in context.skipped

    def test_a_link_out_of_the_repository_is_still_refused(self, repo, tmp_path):
        outside = tmp_path / "outside.py"
        outside.write_text("secret = 1\n", encoding="utf-8")
        (repo / "escape.py").symlink_to(outside)

        paths, context = analyzed(repo)

        assert paths == ["src/app.py"]
        assert ("escape.py", "symbolic link outside repository") in context.skipped

    def test_a_repository_without_links_is_untouched(self, repo):
        (repo / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")

        paths, context = analyzed(repo)

        assert paths == ["docs/guide.md", "src/app.py"]
        assert not any("duplicate" in reason for _, reason in context.skipped)


class TestDownstreamEffects:
    def test_the_dependency_graph_gains_no_phantom_module(self, repo):
        (repo / "src" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (repo / "src" / "app.py").write_text(
            "from helper import VALUE\nprint(VALUE)\n", encoding="utf-8"
        )
        (repo / "src" / "alias.py").symlink_to(repo / "src" / "helper.py")

        documents, _ = TraversalPipeline(folder_path=str(repo)).run()
        graph = build_dependency_graph(
            [
                {"content": doc.page_content, "metadata": dict(doc.metadata)}
                for doc in documents
            ]
        )

        assert not any(node.endswith("alias.py") for node in graph.nodes)

    def test_discover_files_returns_each_path_once(self, repo):
        (repo / "docs" / "app.py").symlink_to(repo / "src" / "app.py")

        discovered, _ = discover_files(str(repo))

        assert len(discovered) == len(set(discovered)) == 1
