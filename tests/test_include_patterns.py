"""``--include`` and ``--exclude`` against paths, not just basenames.

Two things stopped a path-shaped pattern from ever applying during traversal:

* directories were filtered through ``github_file_filter``, so a default-ignored
  directory was pruned from the walk and the files inside it were never reached
  to be judged;
* ``github_file_filter`` relativized the path it was given against ``root_path``
  even when it was already relative, which is what the pipeline passes, so the
  patterns were compared against a path built from the working directory.
"""

import os

import pytest

from repo2readme.loaders.traversal import TraversalPipeline
from repo2readme.utils.filter import (
    github_file_filter,
    include_reaches_into,
    should_descend,
)
from repo2readme.utils.tree import generate_tree

TREE = {
    "src": ["app.py", "secret.py"],
    "src/api": ["routes.py"],
    "dist": ["bundle.js"],
    "dist/nested": ["deep.js"],
    "node_modules/pkg": ["index.js"],
    "node_modules/huge": ["a.js"],
    "vendor": ["lib.go"],
}


@pytest.fixture
def repo(tmp_path):
    for directory, names in TREE.items():
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
        for name in names:
            (tmp_path / directory / name).write_text("const x = 1;\n// code\n")
    return str(tmp_path)


def analyzed(repo, **kwargs):
    documents, _ = TraversalPipeline(folder_path=repo, **kwargs).run()
    return sorted(doc.metadata["relative_path"] for doc in documents)


class TestIncludeReachesIntoIgnoredDirectories:
    def test_a_named_file_inside_a_build_directory_is_analyzed(self, repo):
        assert "dist/bundle.js" in analyzed(
            repo, include_patterns=("dist/bundle.js",)
        )

    def test_a_named_file_inside_node_modules_is_analyzed(self, repo):
        assert "node_modules/pkg/index.js" in analyzed(
            repo, include_patterns=("node_modules/pkg/index.js",)
        )

    def test_a_glob_reaches_through_nested_directories(self, repo):
        assert "dist/nested/deep.js" in analyzed(
            repo, include_patterns=("dist/**/*.js",)
        )

    def test_a_root_only_ignore_can_be_reopened(self, repo):
        assert "vendor/lib.go" in analyzed(
            repo, include_patterns=("vendor/*.go",)
        )

    def test_the_rest_of_the_ignored_directory_stays_ignored(self, repo):
        analyzed_paths = analyzed(
            repo, include_patterns=("node_modules/pkg/index.js",)
        )

        assert "node_modules/huge/a.js" not in analyzed_paths

    def test_a_bare_glob_does_not_reopen_a_build_directory(self, repo):
        # *.js names a file, not a place. Treating it as permission to walk
        # node_modules would undo the default rules for anyone who passes a
        # broad pattern.
        analyzed_paths = analyzed(repo, include_patterns=("*.js",))

        assert not any(
            path.startswith(("dist/", "node_modules/")) for path in analyzed_paths
        )

    def test_nothing_changes_without_an_include(self, repo):
        assert analyzed(repo) == ["src/api/routes.py", "src/app.py", "src/secret.py"]


class TestExcludeMatchesAPath:
    def test_a_path_shaped_exclude_removes_the_file(self, repo):
        assert "src/secret.py" not in analyzed(
            repo, exclude_patterns=("src/secret.py",)
        )

    def test_it_removes_only_that_file(self, repo):
        assert "src/app.py" in analyzed(repo, exclude_patterns=("src/secret.py",))

    def test_a_directory_shaped_exclude_still_works(self, repo):
        analyzed_paths = analyzed(repo, exclude_patterns=("src/api",))

        assert "src/api/routes.py" not in analyzed_paths
        assert "src/app.py" in analyzed_paths

    def test_exclude_wins_over_include(self, repo):
        analyzed_paths = analyzed(
            repo,
            include_patterns=("dist/bundle.js",),
            exclude_patterns=("dist/bundle.js",),
        )

        assert "dist/bundle.js" not in analyzed_paths


class TestFilterTakesRelativePathsToo:
    def test_a_relative_path_is_matched_as_given(self):
        assert github_file_filter(
            "src/secret.py",
            exclude_patterns=("src/secret.py",),
            root_path="/somewhere/else",
            max_file_size_kb=None,
        ) == (False, "excluded by pattern")

    def test_an_absolute_path_is_still_relativized(self):
        assert github_file_filter(
            "/repo/src/secret.py",
            exclude_patterns=("src/secret.py",),
            root_path="/repo",
            max_file_size_kb=None,
        ) == (False, "excluded by pattern")

    def test_both_spellings_agree(self):
        relative = github_file_filter(
            "dist/bundle.js",
            include_patterns=("dist/bundle.js",),
            root_path="/repo",
            max_file_size_kb=None,
        )
        absolute = github_file_filter(
            "/repo/dist/bundle.js",
            include_patterns=("dist/bundle.js",),
            root_path="/repo",
            max_file_size_kb=None,
        )

        assert relative == absolute == (True, "")


class TestShouldDescend:
    @pytest.mark.parametrize(
        "directory, include, expected",
        [
            ("src", (), True),
            ("dist", (), False),
            ("node_modules", (), False),
            ("vendor", (), False),
            ("dist", ("dist/bundle.js",), True),
            ("dist", ("*.py",), False),
            ("dist/nested", ("dist/**/*.js",), True),
            ("vendor", ("vendor/*.go",), True),
            ("vendor/deep", ("vendor/*.go",), False),
            ("node_modules", ("node_modules/pkg/index.js",), True),
            ("node_modules/other", ("node_modules/pkg/index.js",), False),
        ],
    )
    def test_descent_decisions(self, directory, include, expected):
        allowed, _ = should_descend(directory, include_patterns=include)

        assert allowed is expected

    def test_an_excluded_directory_says_why(self):
        assert should_descend("src", exclude_patterns=("src",)) == (
            False,
            "excluded by pattern",
        )

    def test_an_ignored_directory_says_why(self):
        assert should_descend("dist") == (False, "ignored by default rules")

    def test_an_explicitly_included_directory_is_descended(self):
        allowed, _ = should_descend("dist", include_patterns=("dist",))

        assert allowed is True

    def test_the_root_is_not_a_directory_a_pattern_reaches_into(self):
        assert include_reaches_into("", ("anything/at/all",)) is False

    def test_no_patterns_reach_anywhere(self):
        assert include_reaches_into("dist", None) is False
        assert include_reaches_into("dist", ()) is False


class TestTheTreeAgreesWithTheAnalysis:
    def test_an_included_file_appears_in_the_tree(self, repo):
        tree = generate_tree(repo, include_patterns=("dist/bundle.js",))

        assert "bundle.js" in tree

    def test_an_ignored_directory_is_still_absent(self, repo):
        assert "bundle.js" not in generate_tree(repo)

    def test_an_excluded_file_is_absent_from_the_tree(self, repo):
        tree = generate_tree(repo, exclude_patterns=("src/secret.py",))

        assert "secret.py" not in tree
        assert "app.py" in tree


class TestWalkStaysTargeted:
    def test_only_the_named_subdirectory_is_entered(self, repo):
        # Reopening node_modules must not mean walking all of it.
        entered = []
        real_walk = os.walk

        def counting_walk(top, *args, **kwargs):
            for current, dirs, files in real_walk(top, *args, **kwargs):
                entered.append(os.path.relpath(current, repo).replace("\\", "/"))
                yield current, dirs, files

        import repo2readme.loaders.traversal.stages as stages

        original = stages.os.walk
        stages.os.walk = counting_walk
        try:
            analyzed(repo, include_patterns=("node_modules/pkg/index.js",))
        finally:
            stages.os.walk = original

        assert "node_modules/pkg" in entered
        assert "node_modules/huge" not in entered
