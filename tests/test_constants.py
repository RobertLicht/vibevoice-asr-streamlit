"""Baseline tests for module-level constants in vibevoice."""

from vibevoice import (
    AUDIO_EXTENSIONS,
    DEFAULT_CHUNK_SIZE,
    HOP_LENGTH,
    RESAMPLE_DOWN_TO_16K,
    RESAMPLE_DOWN_TO_24K,
    RESAMPLE_DOWN_TO_32K,
    RESAMPLE_KEEP_ORIGINAL,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# AUDIO_EXTENSIONS
# ---------------------------------------------------------------------------


class TestAudioExtensions:
    def test_contains_wav(self):
        assert ".wav" in AUDIO_EXTENSIONS

    def test_contains_mp3(self):
        assert ".mp3" in AUDIO_EXTENSIONS

    def test_contains_flac(self):
        assert ".flac" in AUDIO_EXTENSIONS

    def test_contains_ogg(self):
        assert ".ogg" in AUDIO_EXTENSIONS

    def test_contains_m4a(self):
        assert ".m4a" in AUDIO_EXTENSIONS

    def test_contains_aac(self):
        assert ".aac" in AUDIO_EXTENSIONS

    def test_contains_wma(self):
        assert ".wma" in AUDIO_EXTENSIONS

    def test_excludes_non_audio(self):
        assert ".txt" not in AUDIO_EXTENSIONS
        assert ".pdf" not in AUDIO_EXTENSIONS
        assert ".py" not in AUDIO_EXTENSIONS

    def test_is_frozenset_or_set(self):
        # Should be a set-like type (set or frozenset) for O(1) lookup
        assert isinstance(AUDIO_EXTENSIONS, (set, frozenset))

    def test_all_lowercase(self):
        # Extensions should all be lowercase for consistent comparison
        assert all(ext.islower() for ext in AUDIO_EXTENSIONS if ext)


# ---------------------------------------------------------------------------
# HOP_LENGTH & DEFAULT_CHUNK_SIZE
# ---------------------------------------------------------------------------


class TestHopLength:
    def test_hop_length_is_positive(self):
        assert HOP_LENGTH > 0

    def test_hop_length_equals_expected_3200(self):
        """The hop length should be exactly 3200 per the model spec."""
        assert HOP_LENGTH == 3200


class TestDefaultChunkSize:
    def test_chunk_size_is_multiple_of_hop_length(self):
        assert DEFAULT_CHUNK_SIZE % HOP_LENGTH == 0

    def test_chunk_size_equals_expected_1440000(self):
        """Default chunk = 60 s @ 24 kHz = 1,440,000 samples."""
        assert DEFAULT_CHUNK_SIZE == 1_440_000

    def test_chunk_size_represents_60_seconds_at_sample_rate(self):
        assert DEFAULT_CHUNK_SIZE == SAMPLE_RATE * 60


# ---------------------------------------------------------------------------
# SAMPLE_RATE
# ---------------------------------------------------------------------------


class TestSampleRate:
    def test_sample_rate_equals_expected_24k(self):
        """The model expects 24 kHz audio."""
        assert SAMPLE_RATE == 24_000


# ---------------------------------------------------------------------------
# Resample modes
# ---------------------------------------------------------------------------


class TestResampleModes:
    def test_keep_original_constant(self):
        assert RESAMPLE_KEEP_ORIGINAL == "keep_original"

    def test_down_to_24k_constant(self):
        assert RESAMPLE_DOWN_TO_24K == "down_to_24k"

    def test_down_to_32k_constant(self):
        assert RESAMPLE_DOWN_TO_32K == "down_to_32k"

    def test_down_to_16k_constant(self):
        assert RESAMPLE_DOWN_TO_16K == "down_to_16k"
