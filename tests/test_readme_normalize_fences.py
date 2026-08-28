"""
Tests for what README normalization is allowed to touch (issue #175).

``normalize_markdown`` calls itself "mechanical cleanup only: nothing here
changes the document's meaning", then rewrote every line of the document
without asking whether it was inside a code fence. Trailing whitespace was
stripped and blank runs collapsed inside ``diff`` blocks, fixtures and
expected-output blocks, and Markdown's hard line break was deleted from prose.

Covers:
- a fenced block survives byte for byte
- blank runs inside a fence are not collapsed
- prose keeps its cleanup, including the blank-run limit
- a hard line break survives where it does something, and is stripped where it
  does not
- the fence scanner is the one the validators already use
"""

from __future__ import annotations

import pytest

from repo2readme.readme.postprocess import (
    fenced_flags,
    iter_prose_lines,
    normalize_markdown,
    normalize_prose_line,
    postprocess_readme,
)

# ---------------------------------------------------------------------------
# Fenced content is left alone
# ---------------------------------------------------------------------------


class TestFencedContentSurvives:
    def test_a_diff_block_keeps_its_trailing_spaces(self):
        # A blank context line in a diff hunk is two spaces. Stripping it takes
        # the line out of the hunk.
        text = (
            "# Project\n"
            "\n"
            "```diff\n"
            "- old = 1\n"
            "  \n"
            "+ new = 2\n"
            "```\n"
        )

        assert normalize_markdown(text) == text

    def test_blank_runs_inside_a_fence_are_not_collapsed(self):
        text = "# Project\n\n```python\nfirst()\n\n\n\nlast()\n```\n"

        assert normalize_markdown(text) == text

    def test_a_whole_usage_section_round_trips(self):
        text = (
            "# Project\n"
            "\n"
            "## Usage\n"
            "\n"
            "```diff\n"
            "- old_value = 1\n"
            "+ new_value = 2\n"
            "  trailing spaces here ->   \n"
            "\n"
            "\n"
            "\n"
            "  after three blank lines\n"
            "```\n"
        )

        assert normalize_markdown(text) == text

    def test_a_tilde_fence_is_respected(self):
        text = "# Project\n\n~~~text\nkept   \n\n\n\nkept\n~~~\n"

        assert normalize_markdown(text) == text

    def test_backticks_inside_a_tilde_fence_do_not_close_it(self):
        text = "# Project\n\n~~~markdown\n```\nstill inside   \n```\n~~~\n"

        assert normalize_markdown(text) == text

    def test_an_unclosed_fence_leaves_the_rest_of_the_document_alone(self):
        text = "# Project\n\n```python\nrun()   \n\n\n\nmore()   \n"

        assert normalize_markdown(text) == text


# ---------------------------------------------------------------------------
# Prose still gets cleaned up
# ---------------------------------------------------------------------------


class TestProseIsStillNormalized:
    def test_trailing_whitespace_outside_a_fence_is_removed(self):
        text = "# Title   \n\nSome text.   \n\n```\nkept   \n```\n"

        assert normalize_markdown(text) == "# Title\n\nSome text.\n\n```\nkept   \n```\n"

    def test_blank_runs_outside_a_fence_are_collapsed(self):
        text = "# A\n\n\n\n\n## B\n\n```\n\n\n\n\n```\n"

        assert normalize_markdown(text) == "# A\n\n\n## B\n\n```\n\n\n\n\n```\n"

    def test_leading_and_trailing_blank_lines_are_still_dropped(self):
        assert normalize_markdown("\n\n# A\n\n\n") == "# A\n"

    def test_a_whitespace_only_leading_line_is_dropped(self):
        assert normalize_markdown("   \n# A\n") == "# A\n"

    def test_a_document_that_is_only_a_fence_is_unwrapped_then_normalized(self):
        readme, issues = postprocess_readme("```markdown\n# Title   \n\n\n\n\n```")

        assert readme == "# Title\n"
        assert issues == []


# ---------------------------------------------------------------------------
# Hard line breaks
# ---------------------------------------------------------------------------


class TestHardLineBreaks:
    def test_a_hard_break_mid_paragraph_survives(self):
        text = "# T\n\nLine with a hard break  \ncontinues here.\n"

        assert normalize_markdown(text) == text

    def test_a_longer_run_is_normalised_to_two_spaces(self):
        text = "# T\n\nLine with a hard break     \ncontinues here.\n"

        assert normalize_markdown(text) == "# T\n\nLine with a hard break  \ncontinues here.\n"

    @pytest.mark.parametrize(
        "line,continues,expected",
        [
            ("text  ", True, "text  "),
            ("text  ", False, "text"),
            ("text   ", True, "text  "),
            ("text ", True, "text"),
            ("text\t\t", True, "text"),
            ("## Heading  ", True, "## Heading"),
            ("   ", True, ""),
            ("", True, ""),
        ],
    )
    def test_one_line_at_a_time(self, line, continues, expected):
        assert normalize_prose_line(line, continues=continues) == expected

    def test_a_break_before_a_blank_line_is_stripped(self):
        text = "# T\n\nEnd of paragraph.  \n\nNext paragraph.\n"

        assert normalize_markdown(text) == "# T\n\nEnd of paragraph.\n\nNext paragraph.\n"

    def test_a_break_before_a_code_fence_is_stripped(self):
        text = "# T\n\nRun this:  \n```\ncmd\n```\n"

        assert normalize_markdown(text) == "# T\n\nRun this:\n```\ncmd\n```\n"

    def test_a_break_on_the_last_line_is_stripped(self):
        assert normalize_markdown("# T\n\nThe end.  \n") == "# T\n\nThe end.\n"


# ---------------------------------------------------------------------------
# One scanner, shared with the validators
# ---------------------------------------------------------------------------


class TestFencedFlags:
    def test_the_delimiters_count_as_part_of_the_block(self):
        lines = ["# T", "", "```py", "code", "```", "after"]

        assert fenced_flags(lines) == [False, False, True, True, True, False]

    def test_a_mismatched_marker_does_not_close_the_block(self):
        lines = ["```", "~~~", "```", "after"]

        assert fenced_flags(lines) == [True, True, True, False]

    def test_an_unclosed_fence_runs_to_the_end(self):
        lines = ["# T", "```", "code", "more"]

        assert fenced_flags(lines) == [False, True, True, True]

    def test_no_fences_means_no_flags(self):
        assert fenced_flags(["# T", "text"]) == [False, False]

    def test_empty_input(self):
        assert fenced_flags([]) == []

    def test_prose_lines_agree_with_the_flags(self):
        text = "# T\n\n```py\ncode\n```\n\nafter\n"

        prose = [line for _, line in iter_prose_lines(text)]

        assert prose == ["# T", "", "", "after"]
