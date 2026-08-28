"""
Tests for where an absolute Python import is allowed to resolve (issue #174).

The fallback used to be every directory prefix in the repository, so a bare
name bound to whatever file of that name existed anywhere: ``import utils`` in
one tree found another tree's ``utils.py``, and ``import json`` found a
repository file called ``json.py``. Those edges then ranked files in the
"Core Modules" section the README prompt is built from.

Covers:
- a bare name does not cross into an unrelated directory
- a standard library name does not bind to a same-named repository file
- the imports that were always correct still resolve
- src layouts, flat layouts and namespace packages
- the resulting graph, not just the resolver
"""

from __future__ import annotations

from repo2readme.dependency_graph import (
    PythonRoots,
    _resolve_python_import,
    build_dependency_graph,
    python_roots,
)


def _files(*paths: str) -> dict[str, str]:
    return {path: path for path in paths}


def _doc(path: str, content: str) -> dict:
    return {"content": content, "metadata": {"file_path": path, "file_type": ".py"}}


# ---------------------------------------------------------------------------
# The edges that were invented
# ---------------------------------------------------------------------------


class TestBareNamesDoNotCrossTrees:
    def test_a_bare_name_does_not_reach_a_sibling_tree(self):
        files_map = _files(
            "/repo/backend/utils.py",
            "/repo/backend/app.py",
            "/repo/frontend/scripts/build.py",
        )

        assert (
            _resolve_python_import("/repo/frontend/scripts/build.py", "utils", files_map)
            is None
        )

    def test_a_bare_name_still_resolves_beside_the_importing_file(self):
        files_map = _files("/repo/backend/utils.py", "/repo/backend/app.py")

        assert (
            _resolve_python_import("/repo/backend/app.py", "utils", files_map)
            == "/repo/backend/utils.py"
        )

    def test_a_standard_library_name_does_not_bind_to_a_repository_file(self):
        files_map = _files("/repo/tools/json.py", "/repo/service/handler.py")

        assert (
            _resolve_python_import("/repo/service/handler.py", "json", files_map)
            is None
        )

    def test_a_local_module_shadowing_the_standard_library_still_resolves(self):
        # A json.py beside the importing file really does shadow the standard
        # library at run time, so this edge is not invented.
        files_map = _files("/repo/service/json.py", "/repo/service/handler.py")

        assert (
            _resolve_python_import("/repo/service/handler.py", "json", files_map)
            == "/repo/service/json.py"
        )

    def test_a_bare_name_resolves_at_the_repository_root(self):
        files_map = _files("/repo/settings.py", "/repo/app/views.py")

        assert (
            _resolve_python_import("/repo/app/views.py", "settings", files_map)
            == "/repo/settings.py"
        )


# ---------------------------------------------------------------------------
# Layouts that have to keep working
# ---------------------------------------------------------------------------


class TestLayouts:
    def test_src_layout_resolves_through_the_package_parent(self):
        files_map = _files(
            "/repo/src/mypkg/__init__.py",
            "/repo/src/mypkg/core.py",
            "/repo/src/mypkg/util.py",
            "/repo/tests/test_core.py",
        )

        assert (
            _resolve_python_import("/repo/src/mypkg/core.py", "mypkg.util", files_map)
            == "/repo/src/mypkg/util.py"
        )
        # The test file is outside the package and still finds it, because
        # /repo/src is an import root.
        assert (
            _resolve_python_import("/repo/tests/test_core.py", "mypkg.core", files_map)
            == "/repo/src/mypkg/core.py"
        )

    def test_a_package_at_the_repository_root_resolves(self):
        files_map = _files(
            "/repo/mypkg/__init__.py",
            "/repo/mypkg/core.py",
            "/repo/tests/test_core.py",
        )

        assert (
            _resolve_python_import("/repo/tests/test_core.py", "mypkg.core", files_map)
            == "/repo/mypkg/core.py"
        )

    def test_a_dotted_name_still_finds_a_namespace_package(self):
        # No __init__.py anywhere, so there is nothing to recognise the package
        # by. The dotted name is the only evidence, and it is enough.
        files_map = _files("/repo/src/utils/helpers.py", "/repo/main.py")

        assert (
            _resolve_python_import("/repo/main.py", "utils.helpers", files_map)
            == "/repo/src/utils/helpers.py"
        )

    def test_an_import_naming_the_package_itself_resolves_to_its_init(self):
        files_map = _files("/repo/mypkg/__init__.py", "/repo/run.py")

        assert (
            _resolve_python_import("/repo/run.py", "mypkg", files_map)
            == "/repo/mypkg/__init__.py"
        )

    def test_relative_imports_are_untouched(self):
        files_map = _files(
            "/repo/pkg/__init__.py",
            "/repo/pkg/core.py",
            "/repo/pkg/util.py",
            "/repo/shared.py",
        )

        assert (
            _resolve_python_import("/repo/pkg/core.py", ".util", files_map)
            == "/repo/pkg/util.py"
        )
        assert (
            _resolve_python_import("/repo/pkg/core.py", "..shared", files_map)
            == "/repo/shared.py"
        )


# ---------------------------------------------------------------------------
# Root classification
# ---------------------------------------------------------------------------


class TestRootClassification:
    def test_a_monorepo_gives_each_member_its_own_root(self):
        roots = python_roots(
            _files(
                "/repo/services/a/src/pkg_a/__init__.py",
                "/repo/services/a/src/pkg_a/main.py",
                "/repo/services/b/src/pkg_b/__init__.py",
                "/repo/services/b/src/pkg_b/main.py",
            )
        )

        # Each member's src directory, plus the repository root itself.
        assert roots.import_roots == (
            "/repo",
            "/repo/services/a/src",
            "/repo/services/b/src",
        )
        # Both src directories are import roots, so a member importing another
        # member's package by its real dotted name still resolves - which is
        # what happens when the members are installed side by side. It is the
        # bare, shapeless name that no longer crosses the boundary.
        assert (
            _resolve_python_import(
                "/repo/services/b/src/pkg_b/main.py",
                "pkg_a.main",
                _files(
                    "/repo/services/a/src/pkg_a/__init__.py",
                    "/repo/services/a/src/pkg_a/main.py",
                    "/repo/services/b/src/pkg_b/__init__.py",
                    "/repo/services/b/src/pkg_b/main.py",
                ),
                roots,
            )
            == "/repo/services/a/src/pkg_a/main.py"
        )

    def test_a_data_directory_under_a_package_is_not_a_root(self):
        roots = python_roots(
            _files(
                "/repo/pkg/__init__.py",
                "/repo/pkg/fixtures/nested/sample.py",
            )
        )

        assert "/repo/pkg/fixtures" not in roots.fallback_roots
        assert "/repo/pkg/fixtures/nested" not in roots.fallback_roots

    def test_an_empty_repository_has_no_roots(self):
        roots = python_roots({})

        assert roots == PythonRoots(import_roots=(), fallback_roots=())

    def test_roots_are_derived_when_not_supplied(self):
        files_map = _files("/repo/pkg/__init__.py", "/repo/pkg/core.py", "/repo/run.py")

        supplied = _resolve_python_import(
            "/repo/run.py", "pkg.core", files_map, python_roots(files_map)
        )
        derived = _resolve_python_import("/repo/run.py", "pkg.core", files_map)

        assert supplied == derived == "/repo/pkg/core.py"


# ---------------------------------------------------------------------------
# End to end, through the graph
# ---------------------------------------------------------------------------


class TestGraph:
    def test_invented_edges_no_longer_rank_a_module_as_core(self):
        graph = build_dependency_graph(
            [
                _doc("/repo/backend/utils.py", "X = 1\n"),
                _doc("/repo/backend/app.py", "import utils\n"),
                _doc("/repo/frontend/scripts/build.py", "import utils\n"),
                _doc("/repo/tools/json.py", "Y = 2\n"),
                _doc("/repo/service/handler.py", "import json\nimport os\n"),
            ]
        )

        assert graph.get_dependencies("/repo/backend/app.py") == {
            "/repo/backend/utils.py"
        }
        assert graph.get_dependencies("/repo/frontend/scripts/build.py") == set()
        assert graph.get_dependencies("/repo/service/handler.py") == set()
        assert graph.get_incoming_count("/repo/backend/utils.py") == 1
        assert graph.get_incoming_count("/repo/tools/json.py") == 0

    def test_a_real_package_still_produces_its_edges(self):
        graph = build_dependency_graph(
            [
                _doc("/repo/src/mypkg/__init__.py", ""),
                _doc("/repo/src/mypkg/util.py", "VALUE = 1\n"),
                _doc(
                    "/repo/src/mypkg/core.py",
                    "from mypkg.util import VALUE\nimport os\n",
                ),
                _doc("/repo/tests/test_core.py", "from mypkg import core\n"),
            ]
        )

        assert graph.get_dependencies("/repo/src/mypkg/core.py") == {
            "/repo/src/mypkg/util.py"
        }
        assert graph.get_dependencies("/repo/tests/test_core.py") == {
            "/repo/src/mypkg/__init__.py",
            "/repo/src/mypkg/core.py",
        }
