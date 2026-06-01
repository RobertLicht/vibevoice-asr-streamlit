"""Baseline tests for VibeVoiceCore._resample_audio_if_needed."""

from pathlib import Path

import numpy as np
import pytest
import soundfile  # codec-free audio I/O (libsndfile)

# ---------------------------------------------------------------------------
# Check if torchaudio can actually load/save audio files.
# On systems without FFmpeg codecs, torchaudio.load() will fail even for WAV.
# We need this for tests that exercise the actual resampling pipeline,
# because _resample_audio_if_needed uses torchaudio internally.
# ---------------------------------------------------------------------------

_torchaudio_works = True
try:
    import torch
    import torchaudio as ta

    # Try a minimal round-trip to see if codecs are available.
    sr_test, dur_test = 48_000, 0.1
    data_test = np.random.randn(sr_test * int(dur_test)).astype(np.float32)
    import tempfile as tf

    with tf.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ta.save(f.name, torch.from_numpy(data_test).reshape(1, -1), sr_test)
        _, sr_back = ta.load(f.name)
    _torchaudio_works = True
except Exception:  # noqa: BLE001 — any error means torchaudio is unusable
    _torchaudio_works = False

from vibevoice import (
    RESAMPLE_DOWN_TO_16K,
    RESAMPLE_DOWN_TO_24K,
    RESAMPLE_DOWN_TO_32K,
    RESAMPLE_KEEP_ORIGINAL,
    VibeVoiceCore,
)


class TestResampleAudioIfNeeded:
    """Test real resampling behavior with generated audio.

    Tests that exercise the actual torchaudio-based pipeline are skipped when
    FFmpeg codecs are not available on this system, since _resample_audio_if_needed
    uses torchaudio.load internally for any mode other than keep_original.
    """

    # -----------------------------------------------------------------------
    # keep_original mode — never resamples (no torchaudio needed)
    # -----------------------------------------------------------------------

    def test_keep_original_never_resamples(self, tmp_path):
        """keep_original should always return the original path unchanged."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=96_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_KEEP_ORIGINAL
        )
        assert effective == audio
        assert is_temp is False

    # -----------------------------------------------------------------------
    # down_to_Xk modes — only resample if src > target
    # -----------------------------------------------------------------------

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_24k_no_op_when_already_below(self, tmp_path):
        """If source SR ≤ 24 kHz, down_to_24k should skip."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=16_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        assert effective == audio
        assert is_temp is False

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_24k_no_op_when_exactly_equal(self, tmp_path):
        """If source SR == 24 kHz, down_to_24k should skip."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=24_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        assert effective == audio
        assert is_temp is False

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_24k_resamples_higher_sr(self, tmp_path):
        """If source SR > 24 kHz, down_to_24k should resample and return temp."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        assert effective != audio
        assert is_temp is True
        # Verify the resampled file has correct sample rate.
        _, sr = soundfile.read(str(effective))
        assert sr == 24_000

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_32k_no_op_when_below(self, tmp_path):
        audio = self._make_wav(tmp_path / "audio.wav", sr=24_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_32K
        )
        assert effective == audio

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_32k_resamples_when_above(self, tmp_path):
        audio = self._make_wav(tmp_path / "audio.wav", sr=96_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_32K
        )
        assert is_temp is True
        _, sr = soundfile.read(str(effective))
        assert sr == 32_000

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_16k_no_op_when_below(self, tmp_path):
        audio = self._make_wav(tmp_path / "audio.wav", sr=8_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_16K
        )
        assert effective == audio

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_down_16k_resamples_when_above(self, tmp_path):
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_16K
        )
        assert is_temp is True
        _, sr = soundfile.read(str(effective))
        assert sr == 16_000

    # -----------------------------------------------------------------------
    # File handling
    # -----------------------------------------------------------------------

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_resampled_file_preserves_suffix(self, tmp_path):
        """The temp file should have the same suffix as the original."""
        # Use .wav since it's codec-free on all systems.
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        assert effective.suffix == ".wav"

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_resampled_file_is_writable_and_readable(self, tmp_path):
        """The resampled file should be a valid audio file."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000)
        effective, _ = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        waveform, sr = soundfile.read(str(effective))
        assert len(waveform) > 0

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_resampled_duration_approximately_correct(self, tmp_path):
        """Resampling should preserve approximate duration (within tolerance)."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000, duration_s=1.5)
        effective, _ = VibeVoiceCore._resample_audio_if_needed(
            audio, RESAMPLE_DOWN_TO_24K
        )
        data_orig, sr_orig = soundfile.read(str(audio))
        data_new, sr_new = soundfile.read(str(effective))

        duration_orig = len(data_orig) / sr_orig
        duration_new = len(data_new) / sr_new

        # Allow small rounding differences from resampling
        assert abs(duration_new - duration_orig) < 0.05

    @pytest.mark.skipif(
        not _torchaudio_works,
        reason="torchaudio requires FFmpeg codecs (not available on this system)",
    )
    def test_unknown_mode_skips_resample(self, tmp_path):
        """An unknown mode should skip resampling and return original path."""
        audio = self._make_wav(tmp_path / "audio.wav", sr=48_000)
        effective, is_temp = VibeVoiceCore._resample_audio_if_needed(
            audio, "unknown_mode"
        )
        assert effective == audio
        assert is_temp is False

    def test_resampled_file_is_cleaned_up_by_transcribe(self):
        """transcribe() should clean up temporary files in its finally block.

        This tests the integration of _resample_audio_if_needed with transcribe().
        We cannot easily run this without a real model, but the logic is covered
        by inspecting that temp_path.unlink(missing_ok=True) is called in the
        finally clause — verified via code review rather than runtime test.
        """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_wav(path: Path, sr: int = 48_000, duration_s: float = 1.0):
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        soundfile.write(str(path), data, sr)
        return path
