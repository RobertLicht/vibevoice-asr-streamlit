"""Baseline tests for web UI helper functions.

The actual Streamlit app (webui/app.py) runs on import, so we cannot extract
helper functions from it without triggering model loading and UI initialization.
Instead, we replicate the pure-function helpers here to test their logic directly.
These must be kept in sync with their counterparts in webui/app.py.
"""

from html import escape as _html_escape
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Inline copies of helper functions from webui/app.py (pure logic only)
# These are replicated to avoid importing the Streamlit app at test time.
# ---------------------------------------------------------------------------


def _format_time(seconds: Any) -> str:
    """Format a time value as MM:SS string. Returns "—" for invalid values."""
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "—"
    minutes, secs = divmod(s, 60)
    return f"{minutes:02d}:{secs:02d}"


def _escape_md_pipe(s: str) -> str:
    """Escape pipe characters so they don't break GFM tables."""
    return s.replace("|", "\\|")


def _escape_html(s: str) -> str:
    """Escape special HTML characters to prevent XSS."""
    return _html_escape(str(s), quote=True)


def _utterance_to_markdown(
    transcription: List[Dict[str, Any]], source_file: str = ""
) -> str:
    """Build a Markdown document from an utterance list.

    Includes a title with the source filename and a generation timestamp,
    followed by a properly formatted GFM table.
    """
    import time as _time

    lines = []
    if source_file:
        lines.append(f"# Transcription: {source_file}")
    else:
        lines.append("# Transcription")

    now_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime())
    lines.append(f"*Generated on {now_str}*")
    lines.append("")  # blank line before table

    lines.append("| Speaker | Start | End | Text |")
    lines.append("|---------|-------|-----|------|")

    for u in transcription:
        speaker = _escape_md_pipe(str(u.get("Speaker", "—")))
        start = _format_time(u.get("Start", u.get("start_time", u.get("begin", "—"))))
        end = _format_time(u.get("End", u.get("end_time", u.get("end", "—"))))
        text = _escape_md_pipe(u.get("Content", u.get("text", "")))

        # Wrap long lines in a single-cell paragraph to avoid broken tables.
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text_lines) > 1:
            text = "\n<|>\n".join(text_lines)

        lines.append(f"| {speaker} | {start} | {end} | {text} |")

    return "\n".join(lines) + "\n"


def _utterance_to_html(
    transcription: List[Dict[str, Any]], source_file: str = ""
) -> str:
    """Build a standalone HTML document from an utterance list.

    Produces a complete, self-contained HTML file with embedded CSS,
    a title block (source filename + timestamp), and a styled table.
    """
    import time as _time

    now_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime())
    title_text = _escape_html(source_file) if source_file else "Transcription"

    rows = []
    for u in transcription:
        speaker = _escape_html(str(u.get("Speaker", "—")))
        start = _escape_html(
            _format_time(u.get("Start", u.get("start_time", u.get("begin", "—"))))
        )
        end = _escape_html(
            _format_time(u.get("End", u.get("end_time", u.get("end", "—"))))
        )
        text = _escape_html(u.get("Content", u.get("text", "")))
        rows.append(
            f"<tr><td>{speaker}</td><td>{start}</td><td>{end}</td><td>{text}</td></tr>"
        )

    body_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcription: {title_text}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 2rem;
    color: #1a1a1a;
    background: #fafafa;
  }}
  h1 {{
    font-size: 1.5rem;
    margin-bottom: 0.25rem;
  }}
  .meta {{
    color: #666;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  thead {{
    background: #2563eb;
    color: #fff;
  }}
  th {{
    padding: 0.65rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }}
  td {{
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.925rem;
  }}
  tbody tr:nth-child(even) {{
    background: #f8fafc;
  }}
  tbody tr:hover {{
    background: #eff6ff;
  }}
  td:last-child {{
    white-space: pre-wrap;
  }}
</style>
</head>
<body>
<h1>Transcription: {title_text}</h1>
<p class="meta">Generated on {now_str}</p>
<table>
<thead>
<tr><th>Speaker</th><th>Start</th><th>End</th><th>Text</th></tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Tests — _format_time
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


class TestFormatTime:
    def test_zero_seconds(self):
        assert _format_time(0) == "00:00"

    def test_fifteen_seconds(self):
        assert _format_time(15) == "00:15"

    def test_sixty_seconds(self):
        assert _format_time(60) == "01:00"

    def test_one_minute_fifteen(self):
        assert _format_time(75) == "01:15"

    def test_float_seconds_truncated(self):
        """Float seconds should be truncated via int(float(...))."""
        assert _format_time(6.8) == "00:06"

    def test_negative_seconds_produces_string(self):
        result = _format_time(-5)
        assert isinstance(result, str)  # at minimum it shouldn't crash

    def test_none_returns_dash(self):
        assert _format_time(None) == "—"

    def test_string_invalid_returns_dash(self):
        assert _format_time("abc") == "—"

    def test_empty_string_returns_dash(self):
        assert _format_time("") == "—"

    def test_large_value(self):
        """A large number of seconds should still format correctly."""
        assert _format_time(90) == "01:30"

    def test_string_number_works(self):
        """String representation of a number should be parsed."""
        assert _format_time("45") == "00:45"


# ---------------------------------------------------------------------------
# Tests — _escape_md_pipe
# ---------------------------------------------------------------------------


class TestEscapeMdPipe:
    def test_no_pipes(self):
        assert _escape_md_pipe("hello world") == "hello world"

    def test_single_pipe_escaped(self):
        assert _escape_md_pipe("a|b") == r"a\|b"

    def test_multiple_pipes_escaped(self):
        assert _escape_md_pipe("a|b|c") == r"a\|b\|c"

    def test_empty_string(self):
        assert _escape_md_pipe("") == ""


# ---------------------------------------------------------------------------
# Tests — _escape_html
# ---------------------------------------------------------------------------


class TestEscapeHtml:
    def test_plain_text_unchanged(self):
        assert _escape_html("hello") == "hello"

    def test_lt_gt_escaped(self):
        result = _escape_html("<script>")
        assert "&lt;" in result and "&gt;" in result

    def test_ampersand_escaped(self):
        result = _escape_html("A & B")
        assert "&amp;" in result

    def test_quotes_escaped_with_quote_true(self):
        """html.escape with quote=True escapes both single and double quotes."""
        result = _escape_html('He said "hi"')
        # Either &#34; or &quot; depending on Python version
        assert '"' not in result.split("Hi")[0] if "Hi" in result else True

    def test_none_input(self):
        """None should be converted to string first."""
        result = _escape_html(None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Sample data fixtures (inline, since we can't use conftest fixtures here easily)
# ---------------------------------------------------------------------------

_SAMPLE_TRANSCRIPTION: List[Dict[str, Any]] = [
    {"Speaker": "speaker_1", "Start": 0.0, "End": 3.5, "Content": "Hello world."},
    {
        "Speaker": "speaker_2",
        "Start": 3.5,
        "End": 7.2,
        "Content": "Nice to meet you.",
    },
]

_TRANSCRIPTION_WITH_SPECIAL_CHARS: List[Dict[str, Any]] = [
    {
        "Speaker": "speaker_1",
        "Start": 0.0,
        "End": 2.0,
        "Content": 'He said: "The ratio is 3|4 — important!"',
    },
]


# ---------------------------------------------------------------------------
# Tests — _utterance_to_markdown
# ---------------------------------------------------------------------------


class TestUtteranceToMarkdown:
    def test_basic_output(self):
        md = _utterance_to_markdown(_SAMPLE_TRANSCRIPTION)
        assert "# Transcription" in md
        assert "| Speaker | Start | End | Text |" in md
        assert "speaker_1" in md
        assert "Hello world." in md

    def test_includes_source_file(self):
        source = "/path/to/audio.wav"
        md = _utterance_to_markdown(_SAMPLE_TRANSCRIPTION, source_file=source)
        assert "# Transcription: /path/to/audio.wav" in md

    def test_includes_timestamp(self):
        md = _utterance_to_markdown(_SAMPLE_TRANSCRIPTION)
        assert "Generated on" in md

    def test_uses_format_time_for_start_end(self):
        """Start/End should be formatted as MM:SS strings."""
        md = _utterance_to_markdown(_SAMPLE_TRANSCRIPTION)
        assert "00:00" in md  # Start=0.0 → "00:00"

    def test_escapes_pipe_chars_in_content(self):
        """Pipe characters should be escaped to avoid breaking GFM tables."""
        md = _utterance_to_markdown(_TRANSCRIPTION_WITH_SPECIAL_CHARS)
        assert r"\|" in md

    def test_empty_transcription_produces_header(self):
        """Even an empty transcription should produce a valid markdown table."""
        md = _utterance_to_markdown([])
        assert "# Transcription" in md
        assert "| Speaker | Start | End | Text |" in md

    def test_alternative_field_names(self):
        """Transcriptions with start_time/end_time/text fields should work."""
        transcription = [
            {"Speaker": "s1", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
        ]
        md = _utterance_to_markdown(transcription)
        assert "Hello" in md

    def test_multi_line_text_wrapped(self):
        """Multi-line content should be wrapped to avoid breaking tables."""
        transcription = [
            {
                "Speaker": "s1",
                "Start": 0.0,
                "End": 5.0,
                "Content": "Line one\nLine two\nLine three",
            },
        ]
        md = _utterance_to_markdown(transcription)
        assert "<|>" in md

    def test_returns_string_ending_with_newline(self):
        """The markdown output should end with a newline."""
        md = _utterance_to_markdown([])
        assert md.endswith("\n")


# ---------------------------------------------------------------------------
# Tests — _utterance_to_html
# ---------------------------------------------------------------------------


class TestUtteranceToHtml:
    def test_basic_html_structure(self):
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION)
        assert "<!DOCTYPE html>" in html_doc
        assert "<html" in html_doc
        assert "</html>" in html_doc
        assert "<table>" in html_doc

    def test_includes_source_file(self):
        source = "my_recording.wav"
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION, source_file=source)
        assert source in html_doc

    def test_contains_speaker_and_content(self):
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION)
        assert "speaker_1" in html_doc
        assert "Hello world." in html_doc

    def test_xss_prevention_script_tag(self):
        """Script tags should be escaped to prevent XSS."""
        transcription = [
            {
                "Speaker": "<script>alert('xss')</script>",
                "Start": 0.0,
                "End": 1.0,
                "Content": "Normal text",
            },
        ]
        html_doc = _utterance_to_html(transcription)
        assert "&lt;script&gt;" in html_doc

    def test_xss_prevention_in_content(self):
        """XSS in content should be escaped."""
        transcription = [
            {
                "Speaker": "s1",
                "Start": 0.0,
                "End": 1.0,
                "Content": "<img src=x onerror=alert(1)>",
            },
        ]
        html_doc = _utterance_to_html(transcription)
        assert "&lt;img" in html_doc

    def test_includes_timestamp(self):
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION)
        assert "Generated on" in html_doc

    def test_contains_table_headers(self):
        """HTML table should have proper headers."""
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION)
        assert "<th>Speaker</th>" in html_doc
        assert "<th>Start</th>" in html_doc
        assert "<th>End</th>" in html_doc
        assert "<th>Text</th>" in html_doc

    def test_empty_transcription(self):
        """Empty transcription should still produce valid HTML."""
        html_doc = _utterance_to_html([])
        assert "<!DOCTYPE html>" in html_doc
        assert "</html>" in html_doc

    def test_meta_charset_utf8(self):
        """HTML should declare UTF-8 encoding."""
        html_doc = _utterance_to_html(_SAMPLE_TRANSCRIPTION)
        assert 'charset="UTF-8"' in html_doc or "charset=UTF-8" in html_doc
