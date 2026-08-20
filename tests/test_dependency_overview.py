"""
Tests for the rendered Dependency Overview (issue #131).

Covers:
- get_core_modules is deterministic, including across tied counts
- display names disambiguate files that share a basename
- entry points and isolated files are disjoint, and mean what they say
- the rendered block itself
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from repo2readme.dependency_graph import (
    MAX_LISTED_ENTRY_POINTS,
    DependencyGraph,
    display_names,
    path_suffixes,
)


def _tied_graph(modules: int = 6, importers: int = 2) -> DependencyGraph:
    """Modules with identical incoming counts - the case ordering used to lose."""
    graph = DependencyGraph()
    for index in range(modules):
        for importer in range(importers):
            graph.add_edge(
                f"/repo/app/caller{index}_{importer}.py", f"/repo/lib/mod{index}.py"
            )
    return graph


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestCoreModulesAreDeterministic:
    def test_ties_are_broken_by_path(self):
        core = _tied_graph().get_core_modules(top_n=6)
        assert [path for path, _ in core] == sorted(path for path, _ in core)

    def test_repeated_calls_agree(self):
        graph = _tied_graph()
        assert graph.get_core_modules(top_n=5) == graph.get_core_modules(top_n=5)

    def test_top_n_cut_across_a_tie_is_stable(self):
        """Which five of six tied modules survive must not be a coin flip."""
        top_five = [path for path, _ in _tied_graph().get_core_modules(top_n=5)]
        assert top_five == [f"/repo/lib/mod{index}.py" for index in range(5)]

    def test_count_still_dominates_the_path(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/a.py", "/repo/zzz.py")
        graph.add_edge("/repo/b.py", "/repo/zzz.py")
        graph.add_edge("/repo/c.py", "/repo/aaa.py")

        assert graph.get_core_modules(top_n=2) == [
            ("/repo/zzz.py", 2),
            ("/repo/aaa.py", 1),
        ]

    def test_across_processes_with_different_hash_seeds(self):
        """Set iteration order is seeded per process; the output must not be."""
        script = (
            "from repo2readme.dependency_graph import DependencyGraph\n"
            "g = DependencyGraph()\n"
            "for i in range(6):\n"
            "    for j in range(2):\n"
            "        g.add_edge(f'/repo/app/c{i}_{j}.py', f'/repo/lib/m{i}.py')\n"
            "print([p for p, _ in g.get_core_modules(top_n=5)])\n"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for _ in range(5)
        }
        assert len(outputs) == 1

    def test_empty_graph(self):
        assert DependencyGraph().get_core_modules() == []

    def test_files_with_no_dependents_are_excluded(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/main.py", "/repo/lib.py")
        assert graph.get_core_modules() == [("/repo/lib.py", 1)]


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------


class TestPathSuffixes:
    def test_shortest_first(self):
        assert path_suffixes("a/b/c.py") == ["c.py", "b/c.py", "a/b/c.py"]

    def test_single_component(self):
        assert path_suffixes("main.py") == ["main.py"]

    def test_leading_slash_is_not_a_component(self):
        assert path_suffixes("/repo/app.py") == ["app.py", "repo/app.py"]

    def test_backslashes_are_normalized(self):
        assert path_suffixes("a\\b\\c.py") == path_suffixes("a/b/c.py")


class TestDisplayNames:
    def test_unique_basenames_stay_short(self):
        names = display_names(["/repo/auth/login.py", "/repo/billing/charge.py"])
        assert names["/repo/auth/login.py"] == "login.py"
        assert names["/repo/billing/charge.py"] == "charge.py"

    def test_shared_basenames_are_disambiguated(self):
        paths = ["/repo/auth/index.ts", "/repo/billing/index.ts", "/repo/search/index.ts"]
        names = display_names(paths)
        assert names == {
            "/repo/auth/index.ts": "auth/index.ts",
            "/repo/billing/index.ts": "billing/index.ts",
            "/repo/search/index.ts": "search/index.ts",
        }

    def test_names_are_unique(self):
        paths = ["/a/x/index.ts", "/b/x/index.ts", "/c/index.ts", "/d/main.py"]
        names = display_names(paths)
        assert len(set(names.values())) == len(paths)

    def test_only_the_ambiguous_names_grow(self):
        paths = ["/repo/auth/index.ts", "/repo/billing/index.ts", "/repo/app.py"]
        names = display_names(paths)
        assert names["/repo/app.py"] == "app.py"
        assert names["/repo/auth/index.ts"] == "auth/index.ts"

    def test_deeply_nested_collision(self):
        paths = ["/repo/a/b/c/index.ts", "/repo/x/b/c/index.ts"]
        names = display_names(paths)
        assert names["/repo/a/b/c/index.ts"] == "a/b/c/index.ts"
        assert names["/repo/x/b/c/index.ts"] == "x/b/c/index.ts"

    def test_mapping_is_total_for_duplicates(self):
        names = display_names(["/repo/a.py", "/repo/a.py"])
        assert names == {"/repo/a.py": "a.py"}

    def test_empty_input(self):
        assert display_names([]) == {}

    def test_every_input_gets_a_name(self):
        paths = ["/a/index.ts", "/b/index.ts", "/c/util.py", "main.py"]
        assert set(display_names(paths)) == set(paths)


# ---------------------------------------------------------------------------
# Entry points vs isolated files
# ---------------------------------------------------------------------------


class TestEntryPoints:
    def test_a_file_with_no_edges_is_not_an_entry_point(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/main.py", "/repo/lib.py")
        graph.nodes.add("/repo/scratch.py")

        assert graph.get_entry_points() == ["/repo/main.py"]
        assert graph.get_isolated_files() == ["/repo/scratch.py"]

    def test_the_two_buckets_are_disjoint(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/main.py", "/repo/lib.py")
        for index in range(3):
            graph.nodes.add(f"/repo/script{index}.py")

        assert not set(graph.get_entry_points()) & set(graph.get_isolated_files())

    def test_stats_report_both_without_double_counting(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/main.py", "/repo/lib.py")
        graph.nodes.add("/repo/scratch.py")

        stats = graph.get_dependency_stats()
        assert stats["entry_points"] == 1
        assert stats["isolated_files"] == 1
        assert stats["total_files"] == 3

    def test_a_repository_of_standalone_scripts_has_no_entry_points(self):
        """This section used to list every file in such a repository."""
        graph = DependencyGraph()
        for index in range(5):
            graph.nodes.add(f"/repo/script{index}.py")

        assert graph.get_entry_points() == []
        assert len(graph.get_isolated_files()) == 5

    def test_entry_points_are_sorted(self):
        graph = DependencyGraph()
        for name in ("z", "a", "m"):
            graph.add_edge(f"/repo/{name}.py", "/repo/lib.py")

        assert graph.get_entry_points() == [
            "/repo/a.py",
            "/repo/m.py",
            "/repo/z.py",
        ]

    def test_leaf_modules_are_unaffected(self):
        graph = DependencyGraph()
        graph.add_edge("/repo/main.py", "/repo/lib.py")
        assert graph.get_leaf_modules() == ["/repo/lib.py"]


# ---------------------------------------------------------------------------
# The rendered block
# ---------------------------------------------------------------------------


class TestMarkdownSummary:
    def _packaged_graph(self) -> DependencyGraph:
        graph = DependencyGraph()
        for package in ("auth", "billing", "search"):
            graph.add_edge(f"/repo/{package}/routes.ts", f"/repo/{package}/index.ts")
            graph.add_edge(f"/repo/{package}/service.ts", f"/repo/{package}/index.ts")
        return graph

    def test_no_identical_bullets(self):
        rendered = self._packaged_graph().to_markdown_summary()
        bullets = [line for line in rendered.splitlines() if line.startswith("- ")]
        assert len(bullets) == len(set(bullets))

    def test_core_modules_are_named_by_package(self):
        rendered = self._packaged_graph().to_markdown_summary()
        assert "`auth/index.ts` (2 incoming dependencies)" in rendered
        assert "`billing/index.ts` (2 incoming dependencies)" in rendered

    def test_output_is_byte_identical_across_calls(self):
        graph = self._packaged_graph()
        assert graph.to_markdown_summary() == graph.to_markdown_summary()

    def test_entry_point_wording_matches_the_rule(self):
        rendered = self._packaged_graph().to_markdown_summary()
        assert "import others but are not imported themselves" in rendered

    def test_empty_graph_renders_nothing(self):
        assert DependencyGraph().to_markdown_summary() == ""

    def test_isolated_only_graph_has_no_entry_point_section(self):
        graph = DependencyGraph()
        for index in range(3):
            graph.nodes.add(f"/repo/script{index}.py")

        rendered = graph.to_markdown_summary()
        assert "### Entry Points" not in rendered
        assert "**Isolated files**: 3" in rendered

    def test_entry_points_are_capped_with_a_count(self):
        graph = DependencyGraph()
        for index in range(MAX_LISTED_ENTRY_POINTS + 4):
            graph.add_edge(f"/repo/caller{index:02d}.py", "/repo/lib.py")

        rendered = graph.to_markdown_summary()
        assert "- ... and 4 more" in rendered

    def test_headings_are_unchanged(self):
        rendered = self._packaged_graph().to_markdown_summary()
        for heading in (
            "## Dependency Overview",
            "### Core Modules",
            "### Entry Points",
            "### Dependency Statistics",
        ):
            assert heading in rendered

    @pytest.mark.parametrize("field", [
        "Total files analyzed",
        "Total dependencies",
        "Entry points",
        "Isolated files",
        "Leaf modules",
        "Avg dependencies per file",
    ])
    def test_statistics_fields_are_unchanged(self, field):
        assert field in self._packaged_graph().to_markdown_summary()
