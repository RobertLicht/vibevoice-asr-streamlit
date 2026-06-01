"""VibeVoice ASR — Streamlit Web Interface.

Launch with:

    streamlit run vibevoice_asr/webui/app.py
"""

import html
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Imports from the core module (relative — works when run via `streamlit run`)
# ---------------------------------------------------------------------------
try:
    from vibevoice_asr.vibevoice import (
        DEFAULT_CHUNK_SIZE,
        HOP_LENGTH,
        RESAMPLE_DOWN_TO_16K,
        RESAMPLE_DOWN_TO_24K,
        RESAMPLE_DOWN_TO_32K,
        RESAMPLE_KEEP_ORIGINAL,
        VibeVoiceCore,
    )
except ImportError:
    # Allow direct invocation:  cd vibevoice-asr && streamlit run webui/app.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vibevoice_asr.vibevoice import (
        DEFAULT_CHUNK_SIZE,
        HOP_LENGTH,
        RESAMPLE_DOWN_TO_16K,
        RESAMPLE_DOWN_TO_24K,
        RESAMPLE_DOWN_TO_32K,
        RESAMPLE_KEEP_ORIGINAL,
        VibeVoiceCore,
    )

# ---------------------------------------------------------------------------
# Streamlit page config (set before any other st.* call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VibeVoice ASR", layout="wide")

# Accepted audio extensions for file uploader
_AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"]


# ===================================================================
# Sidebar — Model Configuration
# ===================================================================

with st.sidebar:
    st.header("Model Configuration")

    model_id = st.text_input(
        "Model ID (Hugging Face)",
        value="microsoft/VibeVoice-ASR-HF",
        key="cfg_model_id",
        help=(
            "Hugging Face repository for the VibeVoice ASR model. "
            "Changing this will require re-loading the model."
        ),
    )

    device_choice = st.selectbox(
        "Device",
        options=["auto", "cuda", "cpu"],
        index=0,
        key="cfg_device",
        help=(
            "Where to run inference.\n"
            "  auto — use CUDA if available, otherwise CPU\n"
            "  cuda / cpu — force the selected device."
        ),
    )

    acoustic_chunk_size = st.number_input(
        f"Acoustic Chunk Size (default {DEFAULT_CHUNK_SIZE:,})",
        min_value=HOP_LENGTH,
        value=DEFAULT_CHUNK_SIZE,
        step=HOP_LENGTH,
        key="cfg_chunk",
        help=(
            "Tokeniser chunk size in samples. Must be a multiple of "
            f"{HOP_LENGTH}. Lower values reduce GPU memory but may "
            "introduce extra boundaries."
        ),
    )

    verbose = st.checkbox("Verbose Logging", value=False, key="cfg_verbose")

    # -- Resample mode selector -------------------------------------------
    _RESAMPLE_LABELS: Dict[str, str] = {
        RESAMPLE_KEEP_ORIGINAL: "Keep original",
        RESAMPLE_DOWN_TO_24K: "Downsample to 24 kHz",
        RESAMPLE_DOWN_TO_32K: "Downsample to 32 kHz",
        RESAMPLE_DOWN_TO_16K: "Downsample to 16 kHz",
    }
    _RESAMPLE_OPTIONS = tuple(_RESAMPLE_LABELS.keys())

    resample_mode = st.selectbox(
        "Audio Resampling",
        options=_RESAMPLE_OPTIONS,
        index=_RESAMPLE_OPTIONS.index(RESAMPLE_DOWN_TO_16K),
        format_func=lambda m: _RESAMPLE_LABELS[m],
        key="cfg_resample",
        help=(
            "Controls how the audio sample rate is handled before transcription:\n"
            "  \u2022 **Keep original** \u2014 no resampling; process as-is.\n"
            "  \u2022 **Downsample to 24 kHz** \u2014 resample only if the input's "
            "sample rate exceeds 24 kHz.\n"
            "  \u2022 **Downsample to 16 kHz** \u2014 resample only if the input's "
            "sample rate exceeds 16 kHz.\n\n"
            "Downsampling reduces data volume and processing time for "
            "high-sample-rate audio (e.g. 48 kHz).  Note: the VibeVoice model "
            "is optimized for 24 kHz audio."
        ),
    )

    # -- Local model storage ---------------------------------------------
    st.divider()
    st.subheader("Local Model Storage")

    local_model_dir = st.text_input(
        "Local Model Base Directory",
        value="models",
        key="cfg_local_model_dir",
        help=(
            "Base directory (relative to the project root) where downloaded "
            "models are stored.  The model is loaded from this directory first; "
            "it is downloaded from Hugging Face only if not found locally."
        ),
    )

    # Compute the full local model path
    _model_dir_name = model_id.replace("/", "--")
    _full_local_path = Path(local_model_dir) / _model_dir_name
    _local_model_available = VibeVoiceCore._local_model_exists(_full_local_path)

    # Status indicator
    if local_model_dir:
        if _local_model_available:
            st.success(f"Model found locally: {_full_local_path}")
        else:
            st.info(
                f"Model not found locally at {_full_local_path}. "
                "It will be downloaded from Hugging Face on first use."
            )

        # Download / force-reload buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            if st.button(
                "Download Model",
                key="btn_download_local",
                help="Download the model from Hugging Face to the local directory.",
            ):
                with st.spinner(f"Downloading {model_id} to {_full_local_path} ..."):
                    try:
                        VibeVoiceCore._download_to_local(model_id, local_model_dir)
                        st.success(f"Model downloaded to {_full_local_path}")
                        st.session_state._prev_config = None
                    except Exception as e:
                        st.error(f"Download failed: {e}")

        with col_dl2:
            if st.button(
                "Force Redownload",
                key="btn_force_redownload",
                help="Remove the local copy and re-download from Hugging Face.",
            ):
                with st.spinner(f"Removing and re-downloading {model_id} ..."):
                    try:
                        if _full_local_path.exists():
                            shutil.rmtree(_full_local_path)
                        VibeVoiceCore._download_to_local(model_id, local_model_dir)
                        st.success(f"Model re-downloaded to {_full_local_path}")
                        st.session_state._prev_config = None
                    except Exception as e:
                        st.error(f"Redownload failed: {e}")

    # -- Load / reload trigger -------------------------------------------
    device_value: Optional[str] = None if device_choice == "auto" else device_choice

    def _config_changed() -> bool:
        """Return True if the configuration has changed since last load."""
        prev = st.session_state.get("_prev_config")
        current = (model_id, device_value, acoustic_chunk_size, local_model_dir)
        return prev != current

    if "vibevoice_core" not in st.session_state or _config_changed():
        # Unload the previous model to free GPU/CPU memory before loading a new one.
        if "vibevoice_core" in st.session_state:
            st.warning(
                "Configuration changed — unloading previous model and reloading. "
                "(This may take a moment.)",
                icon="⚠️",
            )
            with st.spinner(text="Unloading previous model ..."):
                st.session_state.vibevoice_core.unload()

        with st.spinner(text=f"Loading {model_id} ..."):
            try:
                core = VibeVoiceCore(
                    model_id=model_id,
                    device=device_value,
                    acoustic_chunk_size=acoustic_chunk_size,
                    verbose=verbose,
                    local_model_dir=local_model_dir or None,
                )
                st.session_state.vibevoice_core = core
                st.session_state._prev_config = (
                    model_id,
                    device_value,
                    acoustic_chunk_size,
                    local_model_dir,
                )
            except Exception as e:  # noqa: BLE001 — surface to user
                st.error(f"Failed to load model: {e}", icon="🚨")
                st.stop()

        # Status badge after loading
        _core = st.session_state.vibevoice_core
        _dtype = str(_core.model.dtype) if hasattr(_core, "model") else "?"
        _dev = _core.device
        st.success(f"Model loaded  •  device: {_dev}  •  dtype: {_dtype}")

    # -- Current config summary -----------------------------------------
    with st.expander("Active Configuration"):
        st.json(
            {
                "model_id": model_id,
                "device": device_value or "auto",
                "acoustic_chunk_size": acoustic_chunk_size,
                "verbose": verbose,
                "resample_mode": _RESAMPLE_LABELS.get(resample_mode, resample_mode),
                "local_model_dir": str(_full_local_path) if local_model_dir else None,
                "local_model_available": _local_model_available,
            }
        )

# ===================================================================
# Helpers
# ===================================================================


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _escape_md_pipe(s: str) -> str:
    """Escape pipe characters so they don't break GFM tables."""
    return s.replace("|", "\\|")


def _utterance_to_markdown(
    transcription: List[Dict[str, Any]], source_file: str = ""
) -> str:
    """Build a Markdown document from an utterance list.

    Includes a title with the source filename and a generation timestamp,
    followed by a properly formatted GFM table.
    """
    lines = []
    if source_file:
        lines.append(f"# Transcription: {source_file}")
    else:
        lines.append("# Transcription")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
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


def _markdown_download(
    transcription: List[Dict[str, Any]],
    source_file: str,
    file_name: str = "transcription.md",
) -> None:
    """Show a download button for Markdown content."""
    md_bytes = _utterance_to_markdown(transcription, source_file).encode("utf-8")
    st.download_button(
        label="⬇ Download Markdown",
        data=md_bytes,
        file_name=file_name,
        mime="text/markdown",
    )


def _escape_html(s: str) -> str:
    """Escape special HTML characters to prevent XSS."""
    return html.escape(str(s), quote=True)


def _utterance_to_html(
    transcription: List[Dict[str, Any]], source_file: str = ""
) -> str:
    """Build a standalone HTML document from an utterance list.

    Produces a complete, self-contained HTML file with embedded CSS,
    a title block (source filename + timestamp), and a styled table.
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
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
# Helpers
# ---------------------------------------------------------------------------


def _format_time(seconds: Any) -> str:
    """Format a time value as MM:SS string. Returns "—" for invalid values."""
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "—"
    minutes, secs = divmod(s, 60)
    return f"{minutes:02d}:{secs:02d}"


def _utterance_table(transcription: List[Dict[str, Any]]) -> None:
    """Render a transcription list as a Streamlit dataframe."""
    rows = []
    for u in transcription:
        # All values must be strings to avoid mixed-type columns that break
        # PyArrow serialization (e.g., int speaker "0" vs. str fallback "—").
        speaker = str(u.get("Speaker", "—"))
        start = _format_time(u.get("Start", u.get("start_time", u.get("begin", "—"))))
        end = _format_time(u.get("End", u.get("end_time", u.get("end", "—"))))
        text = u.get("Content", u.get("text", ""))
        rows.append({"Speaker": speaker, "Start": start, "End": end, "Text": text})

    st.dataframe(rows, width="stretch")


def _json_download(data: Dict[str, Any], file_name: str = "transcription.json") -> None:
    """Show a download button for JSON content."""
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="⬇ Download JSON",
        data=json_bytes,
        file_name=file_name,
        mime="application/json",
    )


def _html_download(
    transcription: List[Dict[str, Any]],
    source_file: str,
    file_name: str = "transcription.html",
) -> None:
    """Show a download button for HTML content."""
    html_str = _utterance_to_html(transcription, source_file)
    st.download_button(
        label="Download HTML",
        data=html_str.encode("utf-8"),
        file_name=file_name,
        mime="text/html",
    )


# ===================================================================
# Main Page
# ===================================================================

st.title("VibeVoice ASR Web UI")

# -- Ensure session-state keys exist --------------------------------
if "_transcription_result" not in st.session_state:
    st.session_state._transcription_result = None
if "_transcription_source" not in st.session_state:
    st.session_state._transcription_source = None
if "_batch_results" not in st.session_state:
    st.session_state._batch_results = {}

tabs = st.tabs(["Single File (Upload)", "Batch (Local Path)"])

# -------------------------------------------------------------------
# Tab 1 — Single File Upload
# -------------------------------------------------------------------
with tabs[0]:
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=_AUDIO_EXTENSIONS,
        label_visibility="visible",
        help="Supported: WAV, MP3, FLAC, OGG, M4A, AAC, WMA",
    )

    # Clear old results when a different file is uploaded
    if (
        uploaded_file is not None
        and st.session_state._transcription_result is not None
        and uploaded_file.name != st.session_state._transcription_source
    ):
        st.session_state._transcription_result = None
        st.session_state._transcription_source = None

    col_hw, col_cs = st.columns([2, 1])
    with col_hw:
        hotwords_upload = st.text_area(
            "Hotwords (optional)",
            placeholder="e.g.  VibeVoice  ASR  Microsoft",
            help="Words or phrases that may appear in the audio and are not "
            "recognised well by default.",
        )

    with col_cs:
        chunk_override = st.number_input(
            f"Chunk Size Override (leave blank for default {DEFAULT_CHUNK_SIZE:,})",
            min_value=HOP_LENGTH,
            value=None,  # type: ignore[arg-type] — None is fine in Streamlit
            step=HOP_LENGTH,
            help=(
                "Override the acoustic chunk size just for this transcription. "
                f"Must be a multiple of {HOP_LENGTH}."
            ),
        )

    st.divider()

    # -- Transcribe button -----------------------------------------------
    if st.button("Transcribe", type="primary"):
        core: VibeVoiceCore = (
            st.session_state.vibevoice_core
        )  # guaranteed after sidebar loads

        if uploaded_file is None:
            st.warning("Please upload an audio file first.")
        else:
            cs_val: Optional[int] = (
                chunk_override
                if chunk_override is not None and chunk_override > 0
                else None
            )
            hw: Optional[str] = hotwords_upload.strip() or None

            # Clear old results before starting a new transcription
            st.session_state._transcription_result = None
            st.session_state._transcription_source = None

            # Write uploaded bytes to a temporary file (processor needs path)
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner("Transcribing ..."):
                    t0 = time.monotonic()
                    result_data: List[Dict[str, Any]] = core.transcribe(
                        audio_path=tmp_path,
                        hotwords=hw,
                        acoustic_chunk_size=cs_val,
                        resample_mode=resample_mode,
                    )
                    elapsed = time.monotonic() - t0

                # Store results in session state
                st.session_state._transcription_result = result_data
                st.session_state._transcription_source = uploaded_file.name

                st.success(
                    f"Transcribed — {len(result_data)} utterance(s) in {elapsed:.1f}s",
                    icon="✅",
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Transcription failed: {e}", icon="🚨")
            finally:
                tmp_path.unlink(missing_ok=True)

    # -- Results display (read from session state) ------------------------
    _result_data = st.session_state._transcription_result
    _source_file = st.session_state._transcription_source
    _has_result = _result_data is not None and _source_file is not None

    if _has_result:
        st.subheader("Result")
        assert _result_data is not None and _source_file is not None
        _utterance_table(_result_data)

        with st.expander("Raw JSON"):
            st.json(
                {
                    "source_file": _source_file,
                    "transcription": _result_data,
                }
            )

    # -- Download buttons (always visible, disabled when no data) ---------
    _disabled = not _has_result
    _dl_name_base = _source_file.rsplit(".", 1)[0] if _source_file else "transcription"

    # Compute download data when result exists
    if _has_result:
        assert _result_data is not None and _source_file is not None
        _json_data = json.dumps(
            {"source_file": _source_file, "transcription": _result_data},
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        _md_data = _utterance_to_markdown(_result_data, _source_file).encode("utf-8")
        _html_data = _utterance_to_html(_result_data, _source_file).encode("utf-8")
    else:
        _json_data = b""
        _md_data = b""
        _html_data = b""

    col_dl, col_dl2, col_dl3 = st.columns(3)
    with col_dl:
        st.download_button(
            label="⬇ Download JSON",
            data=_json_data,
            file_name=f"{_dl_name_base}.json",
            mime="application/json",
            disabled=_disabled,
            key="dl_single_json",
        )
    with col_dl2:
        st.download_button(
            label="⬇ Download Markdown",
            data=_md_data,
            file_name=f"{_dl_name_base}.md",
            mime="text/markdown",
            disabled=_disabled,
            key="dl_single_md",
        )
    with col_dl3:
        st.download_button(
            label="⬇ Download HTML",
            data=_html_data,
            file_name=f"{_dl_name_base}.html",
            mime="text/html",
            disabled=_disabled,
            key="dl_single_html",
        )

# -------------------------------------------------------------------
# Tab 2 — Batch (Local Path)
# -------------------------------------------------------------------
with tabs[1]:
    st.info(
        "This tab transcribes audio files from paths on the **server's** filesystem. "
        "It is useful when running Streamlit locally or inside a container with mounted volumes."
    )

    input_path = st.text_input(
        "Input Path (file or directory)",
        placeholder="/path/to/audio.wav  or  /path/to/audio_dir/",
        help="Path to a single audio file or a directory containing audio files.",
    )

    output_dir = st.text_input(
        "Output Directory",
        placeholder="/tmp/vibevoice_output",
        help="Directory where JSON transcription results will be saved.",
    )

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        recursive = st.checkbox("Recursive Search", value=True)

    with col_b2:
        chunk_override_batch = st.number_input(
            "Chunk Size Override (optional)",
            min_value=HOP_LENGTH,
            value=None,  # type: ignore[arg-type]
            step=HOP_LENGTH,
        )

    hotwords_batch = st.text_area(
        "Hotwords (applied to all files, optional)",
        placeholder="e.g.  VibeVoice  ASR",
    )

    st.divider()

    # -- Transcribe Batch button -----------------------------------------
    if st.button("Transcribe Batch", type="primary"):
        core: VibeVoiceCore = st.session_state.vibevoice_core

        if not input_path.strip():
            st.warning("Please enter an input path.")
        elif not output_dir.strip():
            st.warning("Please enter an output directory.")
        else:
            cs_val_batch: Optional[int] = (
                chunk_override_batch
                if chunk_override_batch is not None and chunk_override_batch > 0
                else None
            )
            hw_batch: Optional[str] = hotwords_batch.strip() or None

            # Clear old batch results
            st.session_state._batch_results = {}

            with st.spinner("Transcribing batch ..."):
                try:
                    out_files = core.transcribe_batch(
                        input_path=input_path,
                        output_dir=output_dir,
                        hotwords=hw_batch,
                        acoustic_chunk_size=cs_val_batch,
                        recursive=recursive,
                        resample_mode=resample_mode,
                    )

                    if not out_files:
                        st.warning("No audio files were found at the given path.")
                    else:
                        # Store parsed results in session state
                        for f in out_files:
                            try:
                                data = json.loads(f.read_text(encoding="utf-8"))
                                st.session_state._batch_results[f.name] = data
                            except Exception as e2:  # noqa: BLE001
                                st.error(f"Could not read {f}: {e2}")

                        st.success(f"Transcribed {len(out_files)} file(s).", icon="✅")

                except Exception as e:  # noqa: BLE001
                    st.error(f"Batch transcription failed: {e}", icon="🚨")

    # -- Batch results display (read from session state) ------------------
    _batch = st.session_state._batch_results
    if _batch:
        st.subheader("Batch Results")
        for idx, (fname, data) in enumerate(_batch.items(), 1):
            src_name = fname.rsplit(".", 1)[0]
            transcription = data.get("transcription", [])
            with st.expander(f"{idx}. **{fname}**", expanded=False):
                _utterance_table(transcription)
                col_bdl, col_bmd, col_bhd = st.columns(3)
                with col_bdl:
                    _json_download(data, f"{src_name}_result.json")
                with col_bmd:
                    _markdown_download(
                        transcription=transcription,
                        source_file=src_name,
                        file_name=f"{src_name}.md",
                    )
                with col_bhd:
                    _html_download(
                        transcription=transcription,
                        source_file=src_name,
                        file_name=f"{src_name}.html",
                    )
    else:
        with st.container():
            st.info("No batch results yet. Transcribe a batch to see results here.")
            # Placeholder disabled download buttons to show they will appear here
            col_bdl, col_bmd, col_bhd = st.columns(3)
            with col_bdl:
                st.download_button(
                    label="⬇ Download JSON",
                    data=b"",
                    file_name="transcription.json",
                    mime="application/json",
                    disabled=True,
                    key="batch_placeholder_json",
                )
            with col_bmd:
                st.download_button(
                    label="⬇ Download Markdown",
                    data=b"",
                    file_name="transcription.md",
                    mime="text/markdown",
                    disabled=True,
                    key="batch_placeholder_md",
                )
            with col_bhd:
                st.download_button(
                    label="⬇ Download HTML",
                    data=b"",
                    file_name="transcription.html",
                    mime="text/html",
                    disabled=True,
                    key="batch_placeholder_html",
                )
