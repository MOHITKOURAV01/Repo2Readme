"""Import forms the dependency graph has to resolve to the right file.

The resolver used to handle ``from . import x`` by looking for a module named
``helper`` in the source directory, a name that came from a test fixture. Every
``from . import <anything>`` arrived as the bare string ``"."`` because the
parser discarded the imported names, so a package with a ``helper.py`` gained a
dependency edge that did not exist, and the module actually imported kept a zero
incoming count and was reported as an entry point.

The fixtures here deliberately avoid the name ``helper`` where the point is that
the real target is found.
"""

from repo2readme.dependency_graph import (
    _imported_names,
    _package_roots,
    _parse_js_imports,
    _parse_python_imports,
    _resolve_python_import,
    build_dependency_graph,
)


def _files(*paths):
    return {path: path for path in paths}


def _doc(path, content, file_type=".py"):
    return {
        "content": content,
        "metadata": {"file_path": path, "file_type": file_type},
    }


# ---------------------------------------------------------------------------
# from ... import <names>
# ---------------------------------------------------------------------------

class TestParseFromImportNames:
    def test_relative_import_emits_the_name_as_a_module_path(self):
        imports = _parse_python_imports("from . import routes")
        assert "." in imports
        assert ".routes" in imports

    def test_several_names_each_get_a_path(self):
        imports = _parse_python_imports("from . import routes, models")
        assert ".routes" in imports
        assert ".models" in imports

    def test_aliases_are_dropped(self):
        imports = _parse_python_imports("from . import models as m")
        assert ".models" in imports
        assert ".m" not in imports

    def test_parent_package_import(self):
        imports = _parse_python_imports("from .. import shared")
        assert "..shared" in imports

    def test_submodule_of_a_relative_package(self):
        imports = _parse_python_imports("from .utils import formatting")
        assert ".utils" in imports
        assert ".utils.formatting" in imports

    def test_absolute_package_names(self):
        imports = _parse_python_imports("from src.utils import formatting")
        assert "src.utils" in imports
        assert "src.utils.formatting" in imports

    def test_multiline_parenthesized_names(self):
        content = "from src.utils import (\n    alpha,\n    beta as b,\n)"
        imports = _parse_python_imports(content)
        assert "src.utils" in imports
        assert "src.utils.alpha" in imports
        assert "src.utils.beta" in imports

    def test_wildcard_is_not_a_module_name(self):
        imports = _parse_python_imports("from .models import *")
        assert ".models" in imports
        assert ".models.*" not in imports

    def test_duplicates_are_collapsed(self):
        content = "from . import routes\nfrom . import routes\n"
        assert _parse_python_imports(content).count(".routes") == 1

    def test_plain_imports_still_work(self):
        imports = _parse_python_imports("import os, sys  # comment")
        assert imports == ["os", "sys"]


class TestImportedNames:
    def test_simple_list(self):
        assert _imported_names("alpha, beta") == ["alpha", "beta"]

    def test_strips_parentheses_and_newlines(self):
        assert _imported_names("(\n alpha,\n beta,\n)") == ["alpha", "beta"]

    def test_drops_aliases_comments_and_wildcards(self):
        assert _imported_names("alpha as a,  # why\n *") == ["alpha"]

    def test_ignores_dotted_or_malformed_entries(self):
        assert _imported_names("alpha, a.b, ") == ["alpha"]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolveRelativeImports:
    def test_sibling_module_is_found_by_name(self):
        files_map = _files("/repo/pkg/__init__.py", "/repo/pkg/routes.py")

        resolved = _resolve_python_import("/repo/pkg/main.py", ".routes", files_map)

        assert resolved == "/repo/pkg/routes.py"

    def test_sibling_package_is_found_by_name(self):
        files_map = _files("/repo/pkg/routes/__init__.py")

        resolved = _resolve_python_import("/repo/pkg/main.py", ".routes", files_map)

        assert resolved == "/repo/pkg/routes/__init__.py"

    def test_bare_dot_resolves_to_the_package_itself(self):
        files_map = _files("/repo/pkg/__init__.py", "/repo/pkg/routes.py")

        resolved = _resolve_python_import("/repo/pkg/main.py", ".", files_map)

        assert resolved == "/repo/pkg/__init__.py"

    def test_bare_dot_without_an_init_resolves_to_nothing(self):
        files_map = _files("/repo/pkg/routes.py")

        assert _resolve_python_import("/repo/pkg/main.py", ".", files_map) is None

    def test_a_module_named_helper_is_no_longer_special(self):
        # The old code hard-coded this name, so any "from . import x" landed on
        # it. Asking for something else must not return it.
        files_map = _files("/repo/pkg/helper.py", "/repo/pkg/routes.py")

        resolved = _resolve_python_import("/repo/pkg/main.py", ".routes", files_map)

        assert resolved == "/repo/pkg/routes.py"

    def test_an_unresolvable_name_returns_none_rather_than_a_guess(self):
        files_map = _files("/repo/pkg/helper.py")

        assert (
            _resolve_python_import("/repo/pkg/main.py", ".missing", files_map) is None
        )

    def test_parent_package_import(self):
        files_map = _files("/repo/shared.py")

        resolved = _resolve_python_import("/repo/pkg/main.py", "..shared", files_map)

        assert resolved == "/repo/shared.py"


class TestPackageRoots:
    def test_collects_every_directory_prefix(self):
        roots = _package_roots(_files("/repo/src/utils/helpers.py"))

        assert roots == ("/repo", "/repo/src", "/repo/src/utils")

    def test_is_sorted_and_deduplicated(self):
        roots = _package_roots(_files("/repo/b/x.py", "/repo/a/y.py", "/repo/a/z.py"))

        assert roots == ("/repo", "/repo/a", "/repo/b")

    def test_precomputed_roots_give_the_same_answer_as_deriving_them(self):
        files_map = _files("/repo/src/utils/helpers.py")

        with_roots = _resolve_python_import(
            "/repo/main.py", "utils.helpers", files_map, _package_roots(files_map)
        )
        without_roots = _resolve_python_import(
            "/repo/main.py", "utils.helpers", files_map
        )

        assert with_roots == without_roots == "/repo/src/utils/helpers.py"


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

class TestParseJsImports:
    def test_side_effect_import(self):
        assert _parse_js_imports("import './polyfill';") == ["./polyfill"]

    def test_side_effect_import_of_a_stylesheet(self):
        assert "./styles.css" in _parse_js_imports('import "./styles.css";')

    def test_named_re_export(self):
        assert "./utils" in _parse_js_imports("export { format } from './utils';")

    def test_star_re_export(self):
        assert "./models" in _parse_js_imports("export * from './models';")

    def test_multiline_binding_list(self):
        content = "import {\n  alpha,\n  beta,\n} from './multi';"

        assert "./multi" in _parse_js_imports(content)

    def test_existing_forms_still_parse(self):
        content = (
            "import config from './config.js';\n"
            "const user = require('./user.js');\n"
            "import('./lazy.js');\n"
        )
        imports = _parse_js_imports(content)

        assert sorted(imports) == ["./config.js", "./lazy.js", "./user.js"]

    def test_duplicates_are_collapsed(self):
        content = "import './a';\nimport { x } from './a';\n"

        assert _parse_js_imports(content) == ["./a"]

    def test_no_imports(self):
        assert _parse_js_imports("const x = 1;") == []


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestGraphFromRelativeImports:
    def test_the_imported_module_gets_the_incoming_edge(self):
        graph = build_dependency_graph([
            _doc("/repo/pkg/__init__.py", ""),
            _doc("/repo/pkg/main.py", "from . import routes\n"),
            _doc("/repo/pkg/routes.py", "ROUTES = []\n"),
            _doc("/repo/pkg/helper.py", "def help_out():\n    pass\n"),
        ])

        assert "/repo/pkg/routes.py" in graph.get_dependencies("/repo/pkg/main.py")
        assert graph.get_incoming_count("/repo/pkg/routes.py") == 1

    def test_no_edge_is_invented_to_the_module_named_helper(self):
        graph = build_dependency_graph([
            _doc("/repo/pkg/main.py", "from . import routes\n"),
            _doc("/repo/pkg/routes.py", "ROUTES = []\n"),
            _doc("/repo/pkg/helper.py", "def help_out():\n    pass\n"),
        ])

        assert "/repo/pkg/helper.py" not in graph.get_dependencies("/repo/pkg/main.py")
        assert graph.get_incoming_count("/repo/pkg/helper.py") == 0

    def test_the_imported_module_is_no_longer_called_an_entry_point(self):
        graph = build_dependency_graph([
            _doc("/repo/pkg/main.py", "from . import routes\n"),
            _doc("/repo/pkg/routes.py", "ROUTES = []\n"),
        ])

        assert graph.get_entry_points() == ["/repo/pkg/main.py"]

    def test_several_names_produce_several_edges(self):
        graph = build_dependency_graph([
            _doc("/repo/pkg/main.py", "from . import routes, models\n"),
            _doc("/repo/pkg/routes.py", "ROUTES = []\n"),
            _doc("/repo/pkg/models.py", "class User:\n    pass\n"),
        ])

        assert graph.get_dependencies("/repo/pkg/main.py") == {
            "/repo/pkg/routes.py",
            "/repo/pkg/models.py",
        }

    def test_importing_a_function_does_not_invent_a_file(self):
        graph = build_dependency_graph([
            _doc("/repo/pkg/main.py", "from .utils import format_name\n"),
            _doc("/repo/pkg/utils.py", "def format_name(x):\n    return x\n"),
        ])

        assert graph.get_dependencies("/repo/pkg/main.py") == {"/repo/pkg/utils.py"}

    def test_js_side_effect_and_barrel_imports_become_edges(self):
        graph = build_dependency_graph([
            _doc("/repo/src/index.js", "import './polyfill';\n", ".js"),
            _doc("/repo/src/polyfill.js", "window.x = 1;\n", ".js"),
            _doc("/repo/src/api.js", "export * from './models';\n", ".js"),
            _doc("/repo/src/models.js", "export const User = {};\n", ".js"),
        ])

        assert graph.get_dependencies("/repo/src/index.js") == {
            "/repo/src/polyfill.js"
        }
        assert graph.get_dependencies("/repo/src/api.js") == {"/repo/src/models.js"}


# ---------------------------------------------------------------------------
# Package resolution, end to end through the loader
# ---------------------------------------------------------------------------
#
# Every candidate _resolve_python_import builds for a package ends in
# __init__.py, and the loader's default ignore rules dropped every one of
# them - so `import pkg` and `from . import x` could not resolve in any real
# repository, however correct the resolver was. These tests go through the
# loader rather than hand-built documents, which is the only way that class of
# regression is visible.


def _graph_for(root):
    from repo2readme.dependency_graph import build_dependency_graph
    from repo2readme.loaders.loader import LocalRepoLoader

    documents, _ = LocalRepoLoader(str(root)).load()
    graph = build_dependency_graph(
        [{"content": d.page_content, "metadata": d.metadata} for d in documents]
    )
    return graph, {d.metadata["relative_path"]: d.metadata["file_path"] for d in documents}


def test_importing_a_package_resolves_to_its_init(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VERSION = '1.0'\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import app\n\nprint(app.VERSION)\n", encoding="utf-8")

    graph, paths = _graph_for(tmp_path)

    assert paths["app/__init__.py"] in graph.get_dependencies(paths["main.py"])


def test_from_dot_import_resolves_to_the_sibling_module(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import routes\n", encoding="utf-8")
    (pkg / "routes.py").write_text("def index():\n    return {}\n", encoding="utf-8")

    graph, paths = _graph_for(tmp_path)

    assert paths["app/routes.py"] in graph.get_dependencies(paths["app/__init__.py"])


def test_from_package_import_module_resolves_through_a_nested_package(tmp_path):
    api = tmp_path / "app" / "api"
    api.mkdir(parents=True)
    (tmp_path / "app" / "__init__.py").write_text(
        "from app.api import routes\n", encoding="utf-8"
    )
    (api / "__init__.py").write_text("from . import routes\n", encoding="utf-8")
    (api / "routes.py").write_text("def index():\n    return {}\n", encoding="utf-8")

    graph, paths = _graph_for(tmp_path)

    deps = graph.get_dependencies(paths["app/__init__.py"])
    assert paths["app/api/routes.py"] in deps


def test_a_package_with_dependents_is_not_reported_as_isolated(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VERSION = '1.0'\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import app\n\nprint(app.VERSION)\n", encoding="utf-8")

    graph, paths = _graph_for(tmp_path)

    assert paths["app/__init__.py"] not in graph.get_isolated_files()
    assert paths["app/__init__.py"] not in graph.get_entry_points()


def test_an_empty_init_does_not_break_resolution_of_its_siblings(tmp_path):
    """The marker file is skipped; the modules beside it still resolve."""
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "routes.py").write_text("from app import models\n", encoding="utf-8")
    (pkg / "models.py").write_text("class Order:\n    pass\n", encoding="utf-8")

    graph, paths = _graph_for(tmp_path)

    assert "app/__init__.py" not in paths
    assert paths["app/models.py"] in graph.get_dependencies(paths["app/routes.py"])
