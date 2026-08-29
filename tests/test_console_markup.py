"""Console output is data, not markup.

``rich.print`` reads square brackets as style tags. Repository paths and
provider error messages are full of square brackets - ``[slug].tsx`` is the
standard spelling of a dynamic route in Next.js, SvelteKit, Nuxt and Remix, and
a provider error routinely carries a JSON fragment - so every value interpolated
into a printed line has to be escaped first.

Two failure modes, and the silent one is the one these tests mostly guard:

* an unknown tag is swallowed, turning ``[slug].tsx`` into ``.tsx`` with no
  warning and no way to tell from the output that it happened;
* a stray closing tag raises ``MarkupError``, which in the failure-report path
  replaces the description of the real failure with a traceback.
"""

import importlib

import pytest
from rich.console import Console

from repo2readme.services.reporting import SummaryFailure, build_report_lines
from repo2readme.utils.console import escaped, styled

# Paths that really appear in real repositories.
ROUTE_PATHS = [
    "pages/posts/[slug].tsx",
    "app/users/[id]/page.tsx",
    "app/[...catchAll]/route.ts",
    "src/routes/[[optional]]/+page.svelte",
    "pages/[category]/[product].vue",
]

# Shapes that make Rich raise rather than swallow.
CLOSING_TAG_TEXTS = [
    "error [/red] here",
    "[/bold]",
    "a [/green] b [/green] c",
]


def render(line: str) -> str:
    """The visible text Rich produces for a line, with styling discarded."""
    console = Console(file=None, record=True, width=200, no_color=True)
    with console.capture() as capture:
        console.print(line)
    return capture.get()


class TestEscaped:
    @pytest.mark.parametrize("path", ROUTE_PATHS)
    def test_a_bracketed_path_survives_rendering(self, path):
        assert path in render(escaped(path))

    @pytest.mark.parametrize("text", CLOSING_TAG_TEXTS)
    def test_a_closing_tag_does_not_raise(self, text):
        assert text in render(escaped(text))

    def test_an_unescaped_bracketed_path_is_swallowed(self):
        # The bug this guards against. Kept as a test so that a future change
        # to Rich's parsing does not quietly make the fix look unnecessary.
        assert "[slug]" not in render("pages/posts/[slug].tsx")

    def test_an_unescaped_closing_tag_raises(self):
        from rich.errors import MarkupError

        with pytest.raises(MarkupError):
            render("error [/red] here")

    def test_accepts_a_non_string(self):
        from pathlib import Path

        assert "[id].tsx" in render(escaped(Path("app/[id].tsx")))
        assert "[404]" in render(escaped(RuntimeError("failed [404]")))

    def test_an_empty_value_stays_empty(self):
        assert escaped("") == ""

    def test_ordinary_text_is_unchanged(self):
        assert escaped("src/main.py") == "src/main.py"

    def test_a_value_that_is_already_escaped_is_not_double_rendered(self):
        # Escaping twice is visible in the output, so the helper must be called
        # exactly once per value; this pins the round trip for one call.
        assert render(escaped("[slug].tsx")).strip() == "[slug].tsx"


class TestStyled:
    def test_the_style_is_applied_as_markup(self):
        console = Console(width=200, force_terminal=True, color_system="truecolor")
        with console.capture() as capture:
            console.print(styled("ok", "green"))
        out = capture.get()
        # The value is there, and it carries an SGR sequence rather than the
        # literal tag, which is what "the style was applied" means.
        assert "ok" in out
        assert "\x1b[" in out

    @pytest.mark.parametrize("path", ROUTE_PATHS)
    def test_the_value_cannot_break_out_of_the_style(self, path):
        assert path in render(styled(path, "green"))

    def test_a_value_containing_the_closing_tag_does_not_raise(self):
        assert "a [/green] b" in render(styled("a [/green] b", "green"))

    def test_the_wrapping_tags_are_not_escaped(self):
        # If the tags were escaped too the user would see them as text.
        assert "[green]" not in render(styled("ok", "green"))


class TestFailureReport:
    def _report(self, failures):
        return build_report_lines(len(failures) + 1, 1, failures)

    def test_a_bracketed_path_reaches_the_user_intact(self):
        failures = [SummaryFailure(file_path="pages/[slug].tsx", reason="rate limit")]
        rendered = "".join(render(line) for line in self._report(failures))
        assert "pages/[slug].tsx" in rendered

    def test_two_bracketed_paths_stay_distinguishable(self):
        # Unescaped, both of these render as ".tsx" and the report names two
        # different files identically.
        failures = [
            SummaryFailure(file_path="pages/[slug].tsx", reason="rate limit"),
            SummaryFailure(file_path="app/[id].tsx", reason="rate limit"),
        ]
        rendered = "".join(render(line) for line in self._report(failures))
        assert "pages/[slug].tsx" in rendered
        assert "app/[id].tsx" in rendered

    def test_a_provider_error_containing_a_closing_tag_does_not_raise(self):
        failures = [
            SummaryFailure(file_path="a.py", reason="Error code: 400 [/red] bad")
        ]
        rendered = "".join(render(line) for line in self._report(failures))
        assert "[/red]" in rendered

    def test_a_provider_error_containing_json_is_shown_verbatim(self):
        reason = "Error code: 429 - {'error': ['rate_limit', 'tpm']}"
        failures = [SummaryFailure(file_path="a.py", reason=reason)]
        rendered = "".join(render(line) for line in self._report(failures))
        assert "['rate_limit', 'tpm']" in rendered

    def test_the_counts_are_still_markup(self):
        failures = [SummaryFailure(file_path="a.py", reason="boom")]
        lines = self._report(failures)
        assert any("[yellow]1 file(s):[/yellow]" in line for line in lines)

    def test_the_happy_path_is_still_quiet(self):
        assert build_report_lines(3, 3, []) == []

    def test_every_line_of_a_report_is_renderable(self):
        failures = [
            SummaryFailure(file_path=f"app/[{n}]/page.tsx", reason=f"boom [/x] {n}")
            for n in range(12)
        ]
        for line in self._report(failures):
            render(line)  # must not raise

    def test_the_overflow_line_is_reached_with_bracketed_paths(self):
        failures = [
            SummaryFailure(file_path=f"app/[{n}].tsx", reason="same reason")
            for n in range(9)
        ]
        rendered = "".join(render(line) for line in self._report(failures))
        assert "... and 4 more" in rendered


class TestCliOutput:
    """The dry-run listing is the command whose whole value is naming files."""

    @pytest.fixture
    def repo(self, tmp_path):
        (tmp_path / "pages" / "posts").mkdir(parents=True)
        (tmp_path / "app" / "users").mkdir(parents=True)
        (tmp_path / "pages" / "posts" / "[slug].tsx").write_text(
            "export default function Post() { return null }\n"
        )
        (tmp_path / "app" / "users" / "[id].tsx").write_text(
            "export default function User() { return null }\n"
        )
        (tmp_path / "index.ts").write_text('export const version = "1.0"\n')
        return tmp_path

    def _dry_run(self, repo, tmp_path, monkeypatch):
        from click.testing import CliRunner

        cli_main = importlib.import_module("repo2readme.cli.main")
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(repo), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        return result.output

    def test_the_file_list_names_the_files(self, repo, tmp_path, monkeypatch):
        output = self._dry_run(repo, tmp_path, monkeypatch)
        assert "pages/posts/[slug].tsx" in output
        assert "app/users/[id].tsx" in output

    def test_the_tree_names_the_files(self, repo, tmp_path, monkeypatch):
        output = self._dry_run(repo, tmp_path, monkeypatch)
        assert "[slug].tsx" in output
        assert "[id].tsx" in output

    def test_no_file_is_reduced_to_its_extension(self, repo, tmp_path, monkeypatch):
        # The exact symptom: both bracketed names collapsed to ".tsx".
        output = self._dry_run(repo, tmp_path, monkeypatch)
        for line in output.splitlines():
            # Drop the tree connectors and the tick, leaving the name itself.
            name = line.strip()
            for lead in ("✓", "└──", "├──", "│"):
                name = name.removeprefix(lead).strip()
            assert name != ".tsx"

    def test_an_ordinary_file_is_unaffected(self, repo, tmp_path, monkeypatch):
        assert "index.ts" in self._dry_run(repo, tmp_path, monkeypatch)


class TestNoUnescapedInterpolation:
    """A guard against the next call site being added without the helper.

    The fix is only durable if new output goes through ``escaped``/``styled``.
    Rather than trusting that, check the source: any ``rprint(f"...")`` in the
    CLI that interpolates something must interpolate an escaped value.
    """

    # Values that come from outside this program: a repository path, an
    # exception, a provider's text. Every one of these must be escaped wherever
    # it is interpolated into a printed line.
    USER_DATA = (
        "rel_path",
        "destination",
        "tree",
        "reason",
        "output_target.path",
        "backup_path_for(",
    )

    def _cli_source(self) -> str:
        import pathlib

        # importlib returns the module; "import repo2readme.cli.main as x"
        # would bind the click Group of the same name re-exported by the
        # package __init__.
        module = importlib.import_module("repo2readme.cli.main")
        return pathlib.Path(module.__file__).read_text()

    def test_every_interpolated_rprint_escapes_its_values(self):
        import re

        source = self._cli_source()

        offenders = []
        for match in re.finditer(r'rprint\(([^\n]*)\)', source):
            call = match.group(1)
            for field in re.findall(r"\{([^{}]+)\}", call):
                if "escaped(" in field or "styled(" in field:
                    continue
                if any(marker in field for marker in self.USER_DATA):
                    offenders.append(field)
                elif re.fullmatch(r"\s*e\s*", field):
                    offenders.append(field)  # a bare exception

        assert not offenders, f"unescaped interpolations in rprint: {offenders}"

    def test_the_guard_would_catch_a_regression(self):
        # The guard is only worth having if it fails on the original code.
        import re

        regressed = 'rprint(f"[red]Failed to load repository: {e}[/red]")'
        fields = re.findall(r"\{([^{}]+)\}", regressed)
        assert any(
            re.fullmatch(r"\s*e\s*", f) and "escaped(" not in f for f in fields
        )

    def test_the_bare_rich_escape_is_no_longer_imported(self):
        # One helper owns the policy; two spellings is how half the call sites
        # ended up unescaped in the first place.
        assert "from rich.markup import escape" not in self._cli_source()


class TestErrorPaths:
    """The error paths are the worst place for a MarkupError.

    Every one of these prints an exception whose text repo2readme does not
    control - it comes from a provider, from git, or from the filesystem - and
    each used to hand that text straight to the markup parser.
    """

    def _invoke(self, args, tmp_path, monkeypatch, **patches):
        from click.testing import CliRunner

        cli_main = importlib.import_module("repo2readme.cli.main")
        monkeypatch.chdir(tmp_path)
        for name, value in patches.items():
            monkeypatch.setattr(cli_main, name, value)
        return CliRunner().invoke(cli_main.main, args)

    def test_a_load_failure_carrying_a_bracketed_path_is_reported(
        self, tmp_path, monkeypatch
    ):
        def exploding_loader(*_args, **_kwargs):
            raise RuntimeError("Folder not found: /repos/[client]/app [/red]")

        result = self._invoke(
            ["run", "--local", str(tmp_path)],
            tmp_path,
            monkeypatch,
            RepoLoader=exploding_loader,
        )

        # The reason survives, and no MarkupError replaced it.
        assert "[client]" in result.output
        assert "MarkupError" not in result.output

    def test_a_readme_failure_containing_brackets_is_reported(
        self, tmp_path, monkeypatch
    ):
        from repo2readme.services.orchestrator import ReadmeGenerationError

        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hello')\n")

        def failing_pipeline(**_kwargs):
            raise ReadmeGenerationError("model returned [nothing] usable")

        result = self._invoke(
            # --force skips the "Proceed?" confirmation, not just the overwrite
            # one, so the run reaches the pipeline.
            ["run", "--local", str(source), "--output", str(tmp_path / "OUT.md"),
             "--force"],
            tmp_path,
            monkeypatch,
            run_pipeline=failing_pipeline,
            setup_api_keys=lambda *_a, **_k: None,
            generate_all_summaries=lambda **_k: ([{"a": 1}], []),
            generate_hierarchical_summaries=lambda **_k: [{"a": 1}],
        )

        assert result.exit_code == 1
        assert "[nothing]" in result.output

    def test_the_saved_path_is_shown_intact(self, tmp_path, monkeypatch):
        source = tmp_path / "repo"
        source.mkdir()
        (source / "main.py").write_text("print('hello')\n")
        destination = tmp_path / "[generated]" / "README.md"
        destination.parent.mkdir()

        result = self._invoke(
            ["run", "--local", str(source), "--output", str(destination), "--force"],
            tmp_path,
            monkeypatch,
            run_pipeline=lambda **_k: "# Title\n\nBody.\n",
            setup_api_keys=lambda *_a, **_k: None,
            generate_all_summaries=lambda **_k: ([{"a": 1}], []),
            generate_hierarchical_summaries=lambda **_k: [{"a": 1}],
        )

        assert result.exit_code == 0, result.output
        assert "[generated]" in result.output


class TestSkipReasons:
    """Skip reasons include loader error strings, which carry paths."""

    def test_a_reason_containing_brackets_survives(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        cli_main = importlib.import_module("repo2readme.cli.main")

        source = tmp_path / "repo"
        source.mkdir()
        (source / "keep.py").write_text("x = 1\n")

        real_loader = cli_main.RepoLoader

        class LoaderWithOddSkip:
            def __init__(self, *args, **kwargs):
                self._inner = real_loader(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def load(self, return_skip_info=False):
                result = self._inner.load(return_skip_info=return_skip_info)
                if not return_skip_info:
                    return result
                documents, root_path, loader_obj, skipped = result
                return (
                    documents,
                    root_path,
                    loader_obj,
                    list(skipped) + [("odd.bin", "unreadable [/red] file")],
                )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli_main, "RepoLoader", LoaderWithOddSkip)

        result = CliRunner().invoke(
            cli_main.main, ["run", "--local", str(source), "--dry-run"]
        )

        assert result.exit_code == 0, result.output
        assert "unreadable [/red] file" in result.output
