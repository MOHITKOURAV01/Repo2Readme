"""
Lightweight dependency-aware context graph for README generation.

Builds an internal graph of file dependencies using lightweight regex parsing
for Python and JavaScript/TypeScript. Supports analysis of entry points,
core modules, isolated files, and leaf modules.

No external dependencies. Gracefully handles syntax errors, missing modules,
unsupported languages, and malformed files.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DependencyGraph:
    """
    Internal dependency graph representing relationships between files.

    Attributes:
        nodes: Set of file paths (absolute, normalized)
        outgoing: Mapping from file -> set of files it imports
        incoming: Mapping from file -> set of files that import it
        errors: List of parsing errors encountered (non-fatal)
    """

    nodes: set[str] = field(default_factory=set)
    outgoing: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    incoming: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    errors: list[str] = field(default_factory=list)

    def add_edge(self, source: str, target: str) -> None:
        """Add a dependency edge from source to target."""
        self.nodes.add(source)
        self.nodes.add(target)
        self.outgoing[source].add(target)
        self.incoming[target].add(source)

    def get_outgoing_count(self, file_path: str) -> int:
        """Return the number of files this file imports."""
        return len(self.outgoing.get(file_path, set()))

    def get_incoming_count(self, file_path: str) -> int:
        """Return the number of files that import this file."""
        return len(self.incoming.get(file_path, set()))

    def get_dependencies(self, file_path: str) -> set[str]:
        """Return files that this file depends on."""
        return self.outgoing.get(file_path, set()).copy()

    def get_dependents(self, file_path: str) -> set[str]:
        """Return files that depend on this file."""
        return self.incoming.get(file_path, set()).copy()

    def get_entry_points(self) -> list[str]:
        """
        Return files with no incoming dependencies (entry points).

        These are files that nothing else imports.
        """
        return sorted([f for f in self.nodes if self.get_incoming_count(f) == 0])

    def get_isolated_files(self) -> list[str]:
        """
        Return files with no dependencies at all (neither incoming nor outgoing).
        """
        return sorted([
            f for f in self.nodes
            if self.get_incoming_count(f) == 0 and self.get_outgoing_count(f) == 0
        ])

    def get_leaf_modules(self) -> list[str]:
        """
        Return files that are imported but don't import anything else.
        These are leaf nodes in the dependency graph.
        """
        return sorted([
            f for f in self.nodes
            if self.get_incoming_count(f) > 0 and self.get_outgoing_count(f) == 0
        ])

    def get_core_modules(self, top_n: int = 10) -> list[tuple[str, int]]:
        """
        Return top-N files most depended on by other modules.

        Filters out entry points (zero incoming dependencies) so that core
        modules are those that actually have dependents.
        Returns list of (file_path, incoming_count) sorted by count descending.
        """
        ranked = sorted(
            [(f, self.get_incoming_count(f)) for f in self.nodes if self.get_incoming_count(f) > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]

    def get_dependency_stats(self) -> dict:
        """Return summary statistics about the dependency graph."""
        total_nodes = len(self.nodes)
        total_edges = sum(len(v) for v in self.outgoing.values())

        if total_nodes == 0:
            return {
                "total_files": 0,
                "total_dependencies": 0,
                "entry_points": 0,
                "isolated_files": 0,
                "leaf_modules": 0,
                "avg_dependencies_per_file": 0.0,
                "max_incoming_count": 0,
            }

        incoming_counts = [self.get_incoming_count(f) for f in self.nodes]
        return {
            "total_files": total_nodes,
            "total_dependencies": total_edges,
            "entry_points": len(self.get_entry_points()),
            "isolated_files": len(self.get_isolated_files()),
            "leaf_modules": len(self.get_leaf_modules()),
            "avg_dependencies_per_file": total_edges / total_nodes if total_nodes > 0 else 0.0,
            "max_incoming_count": max(incoming_counts) if incoming_counts else 0,
        }

    def to_markdown_summary(self) -> str:
        """
        Generate a markdown summary of the dependency graph for README inclusion.

        Returns markdown string with sections for Core Modules, Entry Points,
        and Dependency Statistics.
        """
        stats = self.get_dependency_stats()

        if stats["total_files"] == 0:
            return ""

        lines = ["## Dependency Overview\n"]

        # Core modules
        core = self.get_core_modules(top_n=5)
        if core:
            lines.append("### Core Modules\n")
            lines.append("Files most depended on by other modules:\n")
            for file_path, count in core:
                # Extract just the filename and relative path
                display = file_path.split("/")[-1] if "/" in file_path else file_path
                lines.append(f"- `{display}` ({count} incoming dependencies)\n")
            lines.append("\n")

        # Entry points
        entry_points = self.get_entry_points()
        if entry_points:
            lines.append("### Entry Points\n")
            lines.append("Files with no incoming dependencies:\n")
            for ep in entry_points[:10]:  # Limit to top 10
                display = ep.split("/")[-1] if "/" in ep else ep
                lines.append(f"- `{display}`\n")
            if len(entry_points) > 10:
                lines.append(f"- ... and {len(entry_points) - 10} more\n")
            lines.append("\n")

        # Statistics
        lines.append("### Dependency Statistics\n")
        lines.append(f"- **Total files analyzed**: {stats['total_files']}\n")
        lines.append(f"- **Total dependencies**: {stats['total_dependencies']}\n")
        lines.append(f"- **Entry points**: {stats['entry_points']}\n")
        lines.append(f"- **Isolated files**: {stats['isolated_files']}\n")
        lines.append(f"- **Leaf modules**: {stats['leaf_modules']}\n")
        lines.append(f"- **Avg dependencies per file**: {stats['avg_dependencies_per_file']:.1f}\n")

        return "".join(lines)


# ---------------------------------------------------------------------------
# Lightweight import parsers
# ---------------------------------------------------------------------------

def _parent(directory: str) -> str:
    """The directory containing ``directory``, or ``""`` at the top."""
    return directory.rpartition("/")[0]


def _directories(files_map: dict[str, str]) -> set[str]:
    """Every directory prefix appearing in ``files_map``.

    The prefixes are built by walking up from each file's directory. Building
    them by prepending a separator to each path segment produced "//repo" for
    the leading empty segment of an absolute path, so no candidate ever matched
    and the fallback below never resolved anything.
    """
    directories: set[str] = set()
    for path in files_map:
        directory = _parent(path)
        while directory and directory not in directories:
            directories.add(directory)
            directory = _parent(directory)
    return directories


def _regular_packages(files_map: dict[str, str]) -> set[str]:
    """Directories holding an ``__init__.py``.

    A directory with one is a regular package: its modules are imported through
    the package name, never on their own, so the directory itself is not
    somewhere an import can start from. PEP 420 namespace packages have no
    marker file and are therefore invisible here, which is why the last-resort
    branch in :func:`_resolve_python_import` is kept.
    """
    return {
        _parent(path)
        for path in files_map
        if path.endswith("/__init__.py")
    }


def _inside_package(directory: str, packages: set[str]) -> bool:
    """Whether ``directory`` is a regular package, or sits under one.

    Neither is a place an import can start from: a package's modules are
    reached through the package name, and that stays true however many
    non-package directories sit in between.
    """
    candidate = directory
    while candidate:
        if candidate in packages:
            return True
        candidate = _parent(candidate)
    return False


def _repository_root(directories: set[str]) -> str:
    """The deepest directory that contains every path in the tree."""
    if not directories:
        return ""

    segments = [directory.split("/") for directory in directories]
    shortest = min(segments, key=len)

    common: list[str] = []
    for index, part in enumerate(shortest):
        if all(candidate[index] == part for candidate in segments):
            common.append(part)
        else:
            break

    return "/".join(common)


@dataclass(frozen=True)
class PythonRoots:
    """Where an absolute Python import may be resolved from.

    ``import_roots``
        Directories that could genuinely be on ``sys.path``: the parent of each
        top-level regular package, plus the repository root. ``import mypkg.core``
        works because the directory *holding* ``mypkg`` is on the path, so that
        parent is the root - and a directory that is itself a package never is.

    ``fallback_roots``
        Every remaining directory that is not inside a regular package. Only
        dotted imports are resolved against these, and only after everything
        above has failed. See :func:`_resolve_python_import` for why the branch
        survives at all.
    """

    import_roots: tuple[str, ...] = ()
    fallback_roots: tuple[str, ...] = ()


def python_roots(files_map: dict[str, str]) -> PythonRoots:
    """Classify the directories in ``files_map`` as import roots.

    Computed once per graph build and passed down: derived per import, it
    re-walked every path in the repository for every import statement in every
    file.
    """
    directories = _directories(files_map)
    packages = _regular_packages(files_map)

    roots = {
        _parent(package)
        for package in packages
        if _parent(package) not in packages
    }
    roots.discard("")
    roots.add(_repository_root(directories))
    roots.discard("")

    fallback = {
        directory
        for directory in directories
        if directory not in roots and not _inside_package(directory, packages)
    }

    return PythonRoots(
        import_roots=tuple(sorted(roots)),
        fallback_roots=tuple(sorted(fallback)),
    )


def _module_candidates(root: str, module_path: str) -> tuple[str, str]:
    """The package and the module a dotted name could name under ``root``."""
    relative = module_path.replace(".", "/")
    return f"{root}/{relative}/__init__.py", f"{root}/{relative}.py"


def _resolve_python_import(
    source_file: str,
    import_path: str,
    files_map: dict[str, str],
    roots: PythonRoots | None = None,
) -> Optional[str]:
    """
    Attempt to resolve a Python import to an actual file path.

    Absolute imports are tried against, in order, the importing file's own
    directory, the repository's import roots, and - for dotted names only - any
    remaining directory that is not inside a package.

    That last branch used to be the whole of it: every directory prefix in the
    repository was a candidate, for every import. A bare name then bound to
    whatever file of that name happened to exist anywhere in the tree, so
    ``import utils`` in one service resolved to another service's ``utils.py``
    and ``import json`` resolved to a repository file called ``json.py``. A
    single-segment name carries no path shape to check against and is no longer
    resolved that way.

    A dotted name does carry one - ``a.b.c`` only matches a root that really
    has ``a/b/c.py`` beneath it - and it is the one thing that still finds a
    PEP 420 namespace package, which has no ``__init__.py`` to be recognised
    by. It is kept for that, last and narrowed to directories no package
    encloses.

    Args:
        source_file: Absolute path of the file containing the import
        import_path: The module path being imported (e.g., "os", "utils.helpers")
        files_map: Mapping from normalized file paths to absolute paths
        roots: Precomputed roots for ``files_map``. Derived on demand when
            omitted.

    Returns:
        Resolved absolute file path, or None if not found
    """
    # Standard library / third-party: skip
    if not import_path or import_path.startswith("_"):
        return None

    # Determine source directory for both relative and absolute imports
    source_dir = source_file.rsplit("/", 1)[0] if "/" in source_file else ""
    base_dir = source_dir

    # Handle relative imports
    if import_path.startswith("."):
        # Count leading dots
        dots = 0
        for char in import_path:
            if char == ".":
                dots += 1
            else:
                break
        module_path = import_path[dots:]

        # Navigate up directories by moving to parent (dots - 1) times
        base_dir = source_dir
        for _ in range(max(0, dots - 1)):
            base_dir = base_dir.rsplit("/", 1)[0] if "/" in base_dir else ""

        # "from . import x" carries no module after the dots. The names being
        # imported are turned into their own module paths by
        # _parse_python_imports (". " + "x" -> ".x"), so what is left to resolve
        # here is the package itself, which importing from it does execute.
        if not module_path:
            candidate = f"{base_dir}/__init__.py"
            return candidate if candidate in files_map else None

        # Try relative to the computed base_dir
        for candidate in [
            f"{base_dir}/{module_path.replace('.', '/')}/__init__.py",
            f"{base_dir}/{module_path.replace('.', '/')}.py",
        ]:
            if candidate in files_map:
                return candidate
        return None

    # Absolute import. The importing file's own directory comes first: it is
    # what running the file as a script puts on sys.path, and it cannot reach
    # across the tree into an unrelated one.
    module_path = import_path
    for candidate in _module_candidates(base_dir, module_path):
        if candidate in files_map:
            return candidate

    if roots is None:
        roots = python_roots(files_map)

    for root in roots.import_roots:
        for candidate in _module_candidates(root, module_path):
            if candidate in files_map:
                return candidate

    if "." not in module_path:
        return None

    for root in roots.fallback_roots:
        for candidate in _module_candidates(root, module_path):
            if candidate in files_map:
                return candidate

    return None


def _resolve_js_import(
    source_file: str,
    import_path: str,
    files_map: dict[str, str],
) -> Optional[str]:
    """
    Attempt to resolve a JavaScript/TypeScript import to an actual file path.

    Args:
        source_file: Absolute path of the file containing the import
        import_path: The module path being imported (e.g., "./utils", "lodash")
        files_map: Mapping from normalized file paths to absolute paths

    Returns:
        Resolved absolute file path, or None if not found
    """
    # Skip node_modules and absolute imports
    if not import_path or import_path.startswith("node_modules"):
        return None

    # Only resolve relative imports
    if not import_path.startswith("."):
        return None

    source_dir = source_file.rsplit("/", 1)[0] if "/" in source_file else ""

    # Remove query strings and hashes
    clean_path = import_path.split("?")[0].split("#")[0]
    if clean_path.startswith("./"):
        clean_path = clean_path[2:]

    # Normalize relative segments (resolve '.' and '..' without filesystem access)
    clean_path = _normalize_js_path(clean_path)

    # If path goes above source, compute relative base
    if clean_path.startswith("../"):
        base_dir = source_dir
        while clean_path.startswith("../"):
            clean_path = clean_path[3:]
            base_dir = base_dir.rsplit("/", 1)[0] if "/" in base_dir else ""
        clean_path = f"{base_dir}/{clean_path}" if base_dir else clean_path
    else:
        base_dir = source_dir
        clean_path = f"{base_dir}/{clean_path}" if base_dir else clean_path

    extensions = ["", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]
    index_names = ["index.js", "index.jsx", "index.ts", "index.tsx", "index.mjs", "index.cjs"]

    for ext in extensions:
        candidate = f"{clean_path}{ext}"
        if candidate in files_map:
            return candidate

    for idx in index_names:
        candidate = f"{clean_path}/{idx}"
        if candidate in files_map:
            return candidate

    return None


def _normalize_js_path(path: str) -> str:
    """
    Normalize a JS-style relative path by resolving '.' and '..' segments.
    """
    parts = path.split("/")
    out: list[str] = []
    for part in parts:
        if part == "..":
            if out:
                out.pop()
        elif part and part != ".":
            out.append(part)
    return "/".join(out)


# from <module> import <names>, where <names> may be a parenthesized list
# spanning several lines.
_FROM_IMPORT = re.compile(
    r'^[ \t]*from[ \t]+([^\s]+)[ \t]+import[ \t]*(\([^)]*\)|[^\n#]+)',
    re.MULTILINE,
)


def _imported_names(raw: str) -> list[str]:
    """
    Names from the right-hand side of a ``from ... import ...`` statement.

    Aliases, comments, wildcards and the parentheses of a multi-line list are
    dropped, so ``(helper,  # noqa\\n config as cfg)`` yields
    ``["helper", "config"]``.
    """
    cleaned = raw.strip().removeprefix("(").rstrip(")")

    names: list[str] = []
    for part in cleaned.split(","):
        part = re.sub(r'#.*$', '', part, flags=re.MULTILINE).strip()
        if not part:
            continue
        name = part.split()[0]  # drop "as alias"
        if name == "*" or not re.fullmatch(r'\w+', name):
            continue
        names.append(name)
    return names


def _append_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _parse_python_imports(content: str) -> list[str]:
    """
    Extract Python import paths from file content.

    Returns list of imported module paths.

    For ``from <module> import <names>`` the imported names are emitted as
    module paths of their own, alongside the module itself. Any of those names
    can be a submodule - ``from . import routes`` is the usual way to import a
    sibling - and the module path alone does not say which file that is.
    """
    imports: list[str] = []

    # Match: import module [as alias] [, module ...]
    for match in re.finditer(r'^\s*import\s+([^\n#]+)', content, re.MULTILINE):
        line = match.group(1).strip()
        # Remove inline comments
        line = re.sub(r'#.*$', '', line).strip()
        # Split on commas, handle 'as' aliases
        parts = [p.strip().split()[0] for p in line.split(',') if p.strip()]
        for part in parts:
            _append_unique(imports, part)

    # Match: from module import name [as alias] [, name ...]
    for match in _FROM_IMPORT.finditer(content):
        module_path = match.group(1).strip()
        _append_unique(imports, module_path)

        # "." and ".." already end in the separator; "pkg" and ".pkg" need one.
        prefix = module_path if module_path.endswith(".") else f"{module_path}."
        for name in _imported_names(match.group(2)):
            _append_unique(imports, f"{prefix}{name}")

    return imports


# import X from 'module', including a binding list spread over several lines.
# [^;]*? keeps the match inside one statement.
_JS_IMPORT_FROM = re.compile(
    r'import\s+[^;]*?from\s+["\']([^"\']+)["\']', re.DOTALL
)

# import 'module' - no bindings, imported for its side effects (polyfills,
# stylesheets, registrations).
_JS_SIDE_EFFECT_IMPORT = re.compile(
    r'^[ \t]*import\s+["\']([^"\']+)["\']', re.MULTILINE
)

# export { x } from 'module' / export * from 'module' - the barrel-file pattern,
# which is a dependency in the same way an import is.
_JS_EXPORT_FROM = re.compile(
    r'export\s+[^;]*?from\s+["\']([^"\']+)["\']', re.DOTALL
)

_JS_DYNAMIC_IMPORT = re.compile(r'import\s*\(\s*["\']([^"\']+)["\']\s*\)')

_JS_REQUIRE = re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)')


def _parse_js_imports(content: str) -> list[str]:
    """
    Extract JavaScript/TypeScript import paths from file content.

    Returns list of imported module paths.
    """
    imports: list[str] = []

    for pattern in (
        _JS_IMPORT_FROM,
        _JS_SIDE_EFFECT_IMPORT,
        _JS_EXPORT_FROM,
        _JS_DYNAMIC_IMPORT,
        _JS_REQUIRE,
    ):
        for match in pattern.finditer(content):
            _append_unique(imports, match.group(1).strip())

    return imports


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_dependency_graph(documents: list[dict]) -> DependencyGraph:
    """
    Build a dependency graph from a list of documents.

    Each document is a dict with 'content' and 'metadata' keys.
    Metadata must contain 'file_path' (absolute path).

    Args:
        documents: List of document dicts from the traversal pipeline

    Returns:
        DependencyGraph instance with all detected relationships
    """
    graph = DependencyGraph()

    # Build file path normalization map
    # Key: normalized path (forward slashes), Value: original file_path from metadata
    files_map: dict[str, str] = {}
    for doc in documents:
        file_path = doc.get("metadata", {}).get("file_path", "")
        if file_path:
            normalized = file_path.replace("\\", "/")
            files_map[normalized] = file_path

    # Candidate roots for absolute package imports, walked once for the whole
    # graph rather than once per import statement.
    roots = python_roots(files_map)

    # Process each file
    for doc in documents:
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")
        source_file = metadata.get("file_path", "")

        if not source_file or not content:
            continue

        # Detect language from file type or content
        file_type = metadata.get("file_type", "")
        language = _detect_language_from_ext(file_type)

        if language == "python":
            # Register source node so isolated Python files are tracked
            graph.nodes.add(source_file)
            imports = _parse_python_imports(content)
            resolver = partial(_resolve_python_import, roots=roots)
        elif language in ("javascript", "typescript"):
            # Register source node so isolated JS/TS files are tracked
            graph.nodes.add(source_file)
            imports = _parse_js_imports(content)
            resolver = _resolve_js_import
        else:
            # Unsupported language: skip entirely
            continue

        # Resolve imports to actual files
        normalized_source = source_file.replace("\\", "/")
        for import_path in imports:
            try:
                resolved = resolver(normalized_source, import_path, files_map)
                if resolved and resolved != source_file:
                    graph.add_edge(source_file, resolved)
            except Exception:
                # Never fail on individual import resolution
                pass

    return graph


def _detect_language_from_ext(file_type: str) -> str:
    """Detect language from file extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mjs": "javascript",
        ".cjs": "javascript",
    }
    return ext_map.get(file_type.lower(), "")


# ---------------------------------------------------------------------------
# README enrichment
# ---------------------------------------------------------------------------

def enrich_readme_with_graph(readme: str, graph: DependencyGraph) -> str:
    """
    Enhance an existing README with dependency graph information.

    Appends a "Dependency Overview" section if the graph has meaningful data
    and the section is not already present. This ensures idempotency across
    multiple invocations.

    Args:
        readme: Original README markdown string
        graph: DependencyGraph instance

    Returns:
        Enhanced README markdown string
    """
    summary = graph.to_markdown_summary()

    if not summary:
        return readme

    # Avoid duplicate sections if already present
    if "## Dependency Overview" in readme:
        return readme

    # Append dependency overview at the end
    return readme.rstrip() + "\n\n" + summary
