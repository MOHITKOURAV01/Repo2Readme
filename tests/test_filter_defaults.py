"""Tests for the default ignore rules: the manifest allowlist and the
root-scoped build directory names.

These cover the two failure modes described in the issue: dependency manifests
being dropped by a blanket extension ban, and generic directory names such as
``bin`` or ``pkg`` matching at any depth and deleting source trees.
"""

import pytest

from repo2readme.utils.filter import (
    IGNORE_DIRS,
    NESTED_IGNORE_DIRS,
    ROOT_IGNORE_DIRS,
    classify_default_ignore,
    github_file_filter,
    is_default_ignored,
    is_manifest_file,
)

# ---------------------------------------------------------------------------
# Manifest allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements_test.txt",
        "requirements/base.txt",
        "requirements/production.txt",
        "constraints.txt",
        "package.json",
        "composer.json",
        "tsconfig.json",
        "jsconfig.json",
        "bower.json",
        "deno.json",
        "turbo.json",
        "nx.json",
        "lerna.json",
        "angular.json",
        "nest-cli.json",
        "jsr.json",
        "Pipfile",
        ".env.example",
        ".env.sample",
        ".env.template",
    ],
)
def test_manifest_files_are_not_default_ignored(path):
    assert is_manifest_file(path) is True
    assert is_default_ignored(path) is False
    assert classify_default_ignore(path) is None


@pytest.mark.parametrize(
    "path",
    [
        "requirements.txt",
        "package.json",
        "composer.json",
        ".env.example",
    ],
)
def test_manifest_files_pass_the_public_filter(path):
    # max_file_size_kb=None keeps the check purely name based; the size rule is
    # covered separately by test_manifest_still_respects_the_size_limit.
    allowed, reason = github_file_filter(path, max_file_size_kb=None)
    assert allowed is True
    assert reason == ""


def test_nested_manifests_are_allowed():
    assert github_file_filter(
        "backend/requirements.txt", max_file_size_kb=None
    )[0] is True
    assert github_file_filter(
        "frontend/app/package.json", max_file_size_kb=None
    )[0] is True


def test_manifest_allowlist_is_case_insensitive():
    assert is_manifest_file("Requirements.txt") is True
    assert is_manifest_file("PACKAGE.JSON") is True


def test_lock_files_are_still_excluded():
    """The allowlist must not resurrect the files PROTECTED_LARGE_FILES guards."""
    for lock in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml"):
        assert is_manifest_file(lock) is False
        allowed, reason = github_file_filter(lock, max_file_size_kb=None)
        assert allowed is False
        assert reason == "protected large file"


@pytest.mark.parametrize(
    "env_file",
    [
        ".env",
        ".env.local",
        ".env.production",
        ".env.test",
        # Not in IGNORE_FILES, and their suffix is not in IGNORE_EXTENSIONS
        # either, so these used to pass the filter and reach the model.
        ".env.staging",
        ".env.prod",
        ".env.development.local",
        ".env.production.local",
        ".envrc",
    ],
)
def test_real_env_files_are_excluded(env_file):
    assert is_manifest_file(env_file) is False
    assert is_default_ignored(env_file) is True
    assert classify_default_ignore(env_file) == "ignored_file"


def test_env_files_are_excluded_wherever_they_sit():
    assert classify_default_ignore("config/.env.staging") == "ignored_file"
    assert classify_default_ignore("services/api/.envrc") == "ignored_file"


def test_env_templates_survive_the_family_rule():
    for template in (".env.example", ".env.sample", ".env.template"):
        assert classify_default_ignore(template) is None


def test_a_file_merely_starting_with_env_is_unaffected():
    """The rule is about dotfiles: environment.yml is an ordinary manifest."""
    assert classify_default_ignore("environment.yml") is None
    assert classify_default_ignore("env_loader.py") is None


def test_an_explicit_include_still_admits_an_env_file(tmp_path):
    """--include is evaluated before the default rules, by design."""
    (tmp_path / ".env.staging").write_text("KEY=value", encoding="utf-8")

    allowed, reason = github_file_filter(
        ".env.staging",
        include_patterns=[".env.staging"],
        root_path=str(tmp_path),
    )
    assert allowed is True
    assert reason == ""


def test_ordinary_data_files_are_still_excluded():
    """The allowlist is a list of names, not a blanket lift of the ban."""
    for path in ("fixtures/users.json", "data/export.csv", "notes.txt", "app.log"):
        assert is_default_ignored(path) is True
        assert classify_default_ignore(path) == "ignored_extension"


def test_directory_rules_beat_the_manifest_allowlist():
    """A manifest inside a dependency directory stays ignored."""
    for path in (
        "node_modules/left-pad/package.json",
        "node_modules/pkg/requirements.txt",
        "dist/package.json",
        ".venv/lib/requirements.txt",
    ):
        assert is_default_ignored(path) is True
        assert classify_default_ignore(path) == "build_directory"


# ---------------------------------------------------------------------------
# Root-scoped versus nested directory names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/bin/run.py",
        "app/public/routes.rb",
        "lib/env/loader.py",
        "cmd/vendor/adapter.go",
        "src/out/formatter.ts",
        "app/logs/writer.py",
    ],
)
def test_generic_directory_names_are_allowed_when_nested(path):
    assert is_default_ignored(path) is False
    assert github_file_filter(path, max_file_size_kb=None)[0] is True


@pytest.mark.parametrize(
    "path",
    [
        "bin/activate",
        "public/index.html",
        "vendor/autoload.php",
        "venv/lib/site.py",
        "env/lib/site.py",
        "logs/app.out",
        "out/bundle.js",
    ],
)
def test_generic_directory_names_are_ignored_at_the_root(path):
    assert is_default_ignored(path) is True
    assert classify_default_ignore(path) == "build_directory"


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/pkg/index.js",
        "src/node_modules/pkg/index.js",
        "a/b/c/__pycache__/mod.pyc",
        "frontend/dist/bundle.js",
        "services/api/build/output.js",
        "deep/nested/.git/config",
        "sub/.venv/lib/site.py",
        "x/coverage/report.html",
    ],
)
def test_dependency_directories_are_ignored_at_any_depth(path):
    assert is_default_ignored(path) is True
    assert classify_default_ignore(path) == "build_directory"


def test_egg_info_and_generated_prisma_patterns_still_match():
    assert is_default_ignored("repo2readme.egg-info/PKG-INFO") is True
    assert is_default_ignored("src/generated/prisma/client.ts") is True


@pytest.mark.parametrize(
    "path",
    [
        "pkg/server/main.go",
        "pkg/handlers/auth.go",
        "packages/core/src/index.ts",
        "packages/ui/component.tsx",
    ],
)
def test_go_and_monorepo_layouts_are_no_longer_ignored(path):
    """``pkg/`` and ``packages/`` at the root are source, not build output."""
    assert is_default_ignored(path) is False
    assert github_file_filter(path, max_file_size_kb=None)[0] is True


def test_ignore_dirs_remains_the_union_of_both_sets():
    """Existing callers introspect IGNORE_DIRS; keep it meaningful."""
    assert IGNORE_DIRS == NESTED_IGNORE_DIRS | ROOT_IGNORE_DIRS
    assert "node_modules" in IGNORE_DIRS
    assert "bin" in IGNORE_DIRS
    assert NESTED_IGNORE_DIRS.isdisjoint(ROOT_IGNORE_DIRS)


# ---------------------------------------------------------------------------
# classify_default_ignore
# ---------------------------------------------------------------------------


def test_classify_returns_none_for_ordinary_source_files():
    for path in ("src/main.py", "README.md", "Dockerfile", "pyproject.toml"):
        assert classify_default_ignore(path) is None


def test_classify_distinguishes_the_three_categories():
    assert classify_default_ignore("node_modules/a.js") == "build_directory"
    assert classify_default_ignore("__init__.py") == "ignored_file"
    assert classify_default_ignore("logo.png") == "ignored_extension"


def test_classify_handles_windows_separators():
    assert classify_default_ignore("src\\bin\\run.py") is None
    assert classify_default_ignore("node_modules\\pkg\\index.js") == "build_directory"


def test_explicit_exclude_still_wins_over_a_manifest():
    allowed, reason = github_file_filter(
        "package.json", exclude_patterns=["package.json"], max_file_size_kb=None
    )
    assert allowed is False
    assert reason == "excluded by pattern"


def test_manifest_still_respects_the_size_limit(tmp_path):
    big = tmp_path / "requirements.txt"
    big.write_text("x" * 300 * 1024, encoding="utf-8")

    allowed, reason = github_file_filter(
        "requirements.txt", root_path=str(tmp_path), max_file_size_kb=200
    )
    assert allowed is False
    assert "exceeds maximum file size" in reason
