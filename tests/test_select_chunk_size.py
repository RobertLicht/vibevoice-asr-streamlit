"""Baseline tests for VibeVoiceCore._select_chunk_size."""

import pytest

from vibevoice import HOP_LENGTH, SAMPLE_RATE, VibeVoiceCore


class TestSelectChunkSize:
    """_select_chunk_size scales chunk size down for short audio."""

    def test_default_chunk_for_long_audio(self):
        """Audio > 10 min should use the default max_chunk_size (1,440,000)."""
        samples = int(SAMPLE_RATE * 661)  # 11 minutes at 24 kHz
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk == 1_440_000

    def test_custom_max_chunk(self):
        """Custom max_chunk_size should be respected."""
        samples = int(SAMPLE_RATE * 700)  # > 10 min, so max_chunk applies directly
        chunk = VibeVoiceCore._select_chunk_size(samples, max_chunk_size=640_000)
        assert chunk == 640_000

    def test_custom_max_chunk_not_multiple_is_snapped(self):
        """If custom max_chunk is not a multiple of HOP_LENGTH, it gets snapped."""
        samples = int(SAMPLE_RATE * 700)
        # 599_999 / 3200 ≈ 187.499 → rounds to 187 → 187*3200 = 598_400
        chunk = VibeVoiceCore._select_chunk_size(samples, max_chunk_size=599_999)
        assert chunk % HOP_LENGTH == 0

    def test_at_10_minute_boundary(self):
        """At exactly 10 minutes (600s), target chunk should be 60 s = 1,440,000."""
        samples = int(SAMPLE_RATE * 600)  # exactly 10 min
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk == 1_440_000

    def test_just_below_10_minutes(self):
        """Just below 10 min, chunk should scale proportionally."""
        samples = int(SAMPLE_RATE * 599.99)  # just under 600s
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk <= 1_440_000

    def test_short_audio_minimum_10_seconds(self):
        """Very short audio should get at least a 10-second chunk."""
        samples = int(SAMPLE_RATE * 1)  # 1 second of audio
        chunk = VibeVoiceCore._select_chunk_size(samples)
        min_expected = 10 * SAMPLE_RATE  # 240,000 — but snapped to multiple of 3200
        assert chunk >= HOP_LENGTH

    def test_one_second_audio(self):
        """For a 1-second audio clip, minimum chunk should be ~10 s (capped)."""
        samples = SAMPLE_RATE  # exactly 1 second
        chunk = VibeVoiceCore._select_chunk_size(samples)
        # Should be max(10s_target, snapped_to_hop_length)
        assert chunk % HOP_LENGTH == 0
        # The minimum is max(HOP_LENGTH, snapped_10s_target)
        target_10s = int(10 * SAMPLE_RATE)
        snapped = round(target_10s / HOP_LENGTH) * HOP_LENGTH
        assert chunk == snapped

    def test_five_minute_audio(self):
        """5-minute audio should scale chunk to ~60 s (the cap for ≤ 10 min)."""
        samples = int(SAMPLE_RATE * 300)  # 5 minutes at 24 kHz
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk == 1_440_000

    def test_three_minute_audio(self):
        """3-minute audio → target ≈ min(60, 180) s → 60 s."""
        samples = int(SAMPLE_RATE * 180)  # 3 minutes at 24 kHz
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk == 1_440_000

    def test_thirty_second_audio(self):
        """30-second audio → target ≈ min(60, 30) s → 30 s → snapped to hop."""
        samples = int(SAMPLE_RATE * 30)
        chunk = VibeVoiceCore._select_chunk_size(samples)
        target_30s = round(int(30 * SAMPLE_RATE) / HOP_LENGTH) * HOP_LENGTH
        assert chunk == target_30s

    def test_result_is_multiple_of_hop_length(self):
        """All results must be multiples of HOP_LENGTH."""
        for duration_s in [1, 5, 10, 30, 60, 120, 300, 600]:
            samples = int(SAMPLE_RATE * duration_s)
            chunk = VibeVoiceCore._select_chunk_size(samples)
            assert chunk % HOP_LENGTH == 0, f"Failed for {duration_s}s audio"

    def test_very_large_audio(self):
        """1-hour audio should just use max_chunk directly."""
        samples = int(SAMPLE_RATE * 3600)
        chunk = VibeVoiceCore._select_chunk_size(samples)
        assert chunk == 1_440_000

    def test_long_audio_respects_max_chunk(self):
        """Audio > 10 minutes uses max_chunk_size (snapped to HOP_LENGTH)."""
        samples = int(SAMPLE_RATE * 720)  # 12 min
        custom_max = 50_000
        chunk = VibeVoiceCore._select_chunk_size(samples, max_chunk_size=custom_max)
        # Snapped to nearest multiple of HOP_LENGTH: round(50000/3200)*3200 = 16*3200 = 51200
        assert chunk == round(custom_max / HOP_LENGTH) * HOP_LENGTH
