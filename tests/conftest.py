"""Shared test fixtures for VibeVoice ASR baseline tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile  # codec-free audio I/O (libsndfile)

# ---------------------------------------------------------------------------
# Temporary directory fixture — isolated per test
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_audio_dir(tmp_path):
    """Create a temporary directory populated with synthetic audio files.

    Uses soundfile for real WAV files (codec-free via libsndfile). Other
    extensions are placeholder files — discover_audio_files is extension-based
    and does not need actual audio content to validate discovery logic.
    """

    # Helper: write a tiny 1-second mono sine wave at the given sample rate.
    def _write_wav(name, sr=48_000, duration_s=1.0):
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        soundfile.write(str(tmp_path / name), data, sr)

    # Real audio files (WAV only — codec-free via libsndfile)
    _write_wav("test.wav", sr=48_000)

    # Placeholder files with other extensions for discovery tests.
    # Content is irrelevant; discover_audio_files checks extension only.
    (tmp_path / "short.flac").touch()
    (tmp_path / "high_sr.ogg").touch()

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.m4a").touch()

    # A non-audio file (should be ignored)
    (tmp_path / "readme.txt").write_text("not audio")

    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures for mocked VibeVoiceCore instances
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Return a minimal mock model with device and dtype attributes."""
    model = MagicMock()
    model.device = "cpu"
    model.dtype = np.float32
    return model


@pytest.fixture
def mock_processor():
    """Return a minimal mock processor with apply_transcription_request + decode."""

    def _apply_transcription_request(audio=None, prompt=""):
        # Return something that looks like model inputs
        from unittest.mock import MagicMock as M

        result = M()
        result.to.return_value = {
            "input_ids": np.array([[0, 1, 2]]),
        }
        return result

    def _decode(generated_ids, return_format="parsed"):
        return [
            [
                {
                    "Speaker": "speaker_1",
                    "Start": 0.0,
                    "End": 3.5,
                    "Content": "Hello world.",
                },
                {
                    "Speaker": "speaker_2",
                    "Start": 3.5,
                    "End": 7.2,
                    "Content": "Nice to meet you.",
                },
            ]
        ]

    processor = MagicMock()
    processor.apply_transcription_request.side_effect = _apply_transcription_request
    processor.decode.side_effect = _decode
    return processor


@pytest.fixture
def mock_vibevoice_core(mock_model, mock_processor):
    """Create a VibeVoiceCore instance with model and processor patched in.

    This patches the class-level attributes so that ``is_loaded`` returns True
    without actually loading the real model from Hugging Face.
    """
    from vibevoice import VibeVoiceCore

    with patch.object(VibeVoiceCore, "_load_model", return_value=None):
        core = VibeVoiceCore.__new__(VibeVoiceCore)
        core.model_id = "microsoft/VibeVoice-ASR-HF"
        core.device = "cpu"
        core.acoustic_chunk_size = 1_440_000
        core.verbose = False
        core.local_model_dir = None
        core._loaded = True

        # Inject mocked class-level state.
        VibeVoiceCore._model = mock_model
        VibeVoiceCore._processor = mock_processor
        VibeVoiceCore._model_loaded = True
        VibeVoiceCore._ref_count = 1

        yield core

    # Cleanup: restore class-level state to None so later tests are clean.
    VibeVoiceCore._model = None
    VibeVoiceCore._processor = None
    VibeVoiceCore._model_loaded = False
    VibeVoiceCore._ref_count = 0


# ---------------------------------------------------------------------------
# Sample transcription data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transcription():
    """Return a typical parsed transcription result."""
    return [
        {"Speaker": "speaker_1", "Start": 0.0, "End": 3.5, "Content": "Hello world."},
        {
            "Speaker": "speaker_2",
            "Start": 3.5,
            "End": 7.2,
            "Content": "Nice to meet you.",
        },
    ]


@pytest.fixture
def transcription_with_special_chars():
    """Transcription containing characters that need escaping."""
    return [
        {
            "Speaker": "speaker_1",
            "Start": 0.0,
            "End": 2.0,
            "Content": 'He said: "The ratio is 3|4 — important!"',
        },
    ]
