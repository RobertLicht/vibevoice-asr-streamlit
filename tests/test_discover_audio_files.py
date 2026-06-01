"""Baseline tests for VibeVoiceCore.discover_audio_files."""

import pytest

from vibevoice import AUDIO_EXTENSIONS, VibeVoiceCore


class TestDiscoverAudioFiles:
    """discover_audio_files finds audio files by extension."""

    def test_single_file(self, tmp_audio_dir):
        wav = tmp_audio_dir / "test.wav"
        result = VibeVoiceCore.discover_audio_files(wav)
        assert result == [wav]

    def test_directory_recursive_finds_all(self, tmp_audio_dir):
        """Recursive search should find audio files in subdirectories."""
        result = VibeVoiceCore.discover_audio_files(tmp_audio_dir, recursive=True)
        names = {f.name for f in result}
        assert "test.wav" in names
        assert "short.flac" in names
        assert "high_sr.ogg" in names
        assert "nested.m4a" in names

    def test_directory_non_recursive_skips_subdirs(self, tmp_audio_dir):
        """Non-recursive should not find files in subdirectories."""
        result = VibeVoiceCore.discover_audio_files(tmp_audio_dir, recursive=False)
        names = {f.name for f in result}
        assert "test.wav" in names
        assert "nested.m4a" not in names

    def test_excludes_non_audio(self, tmp_audio_dir):
        """Non-audio files (readme.txt) should be excluded."""
        result = VibeVoiceCore.discover_audio_files(tmp_audio_dir)
        suffixes = {f.suffix.lower() for f in result}
        assert ".txt" not in suffixes

    def test_nonexistent_path_raises(self):
        """A non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VibeVoiceCore.discover_audio_files("/no/such/path")

    def test_unknown_extension_in_file_mode_raises(self, tmp_path):
        """Passing a single file with an unknown extension raises ValueError."""
        bad = tmp_path / "data.csv"
        bad.touch()
        with pytest.raises(ValueError, match="Unknown audio extension"):
            VibeVoiceCore.discover_audio_files(bad)

    def test_case_insensitive_extension(self, tmp_audio_dir):
        """Uppercase extensions should still be recognized."""
        upper_wav = tmp_audio_dir / "UPPER.WAV"
        upper_wav.touch()
        result = VibeVoiceCore.discover_audio_files(tmp_audio_dir)
        assert any(f.name == "UPPER.WAV" for f in result)

    def test_empty_directory(self, tmp_path):
        """An empty directory returns an empty list."""
        result = VibeVoiceCore.discover_audio_files(tmp_path)
        assert result == []

    def test_results_sorted(self, tmp_path):
        """Returned paths should be sorted alphabetically."""
        for ext in AUDIO_EXTENSIONS:
            (tmp_path / f"z{ext}").touch()
            (tmp_path / f"a{ext}").touch()
        result = VibeVoiceCore.discover_audio_files(tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_all_returned_paths_are_files(self, tmp_audio_dir):
        """All returned items should be files (not directories)."""
        result = VibeVoiceCore.discover_audio_files(tmp_audio_dir)
        for p in result:
            assert p.is_file()
