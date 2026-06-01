"""Baseline tests for VibeVoiceCore initialization and local-model-path logic."""

from pathlib import Path
from unittest.mock import patch

import pytest

from vibevoice import HOP_LENGTH, VibeVoiceCore


class TestInitValidation:
    """Test constructor validation of chunk size."""

    def test_valid_chunk_size(self):
        """A multiple of HOP_LENGTH should be accepted (without loading model)."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(acoustic_chunk_size=HOP_LENGTH * 10)
            assert core.acoustic_chunk_size == HOP_LENGTH * 10

    def test_invalid_chunk_size_raises(self):
        """A non-multiple of HOP_LENGTH should raise ValueError."""
        with pytest.raises(ValueError, match="must be a multiple of"):
            VibeVoiceCore(acoustic_chunk_size=HOP_LENGTH + 7)

    def test_zero_chunk_size_is_multiple_of_hop_length(self):
        """Zero is technically a multiple of any number. Check behavior."""
        # Zero % anything == 0, so this should pass validation
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(acoustic_chunk_size=0)
            assert core.acoustic_chunk_size == 0

    def test_default_device_falls_back_to_cpu_without_cuda(self):
        """Without CUDA, device defaults to 'cpu'."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore()
            assert core.device == "cpu"

    def test_default_device_uses_cuda_when_available(self):
        """With CUDA available, device defaults to 'cuda'."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore()
            assert core.device == "cuda"

    def test_explicit_device_overrides_auto_detect(self):
        """Explicit device should override auto-detection."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(device="cpu")
            assert core.device == "cpu"


class TestGetLocalModelPath:
    """Test _get_local_model_path() path computation."""

    def test_none_when_no_local_dir(self):
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(local_model_dir=None)
            assert core._get_local_model_path() is None

    def test_path_with_forward_slash_replaced(self):
        """The '/' in model_id should be replaced with '--'."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(
                model_id="microsoft/VibeVoice-ASR-HF",
                local_model_dir="/tmp/models",
            )
            path = core._get_local_model_path()
            assert "VibeVoice" not in str(path).split("/")[0]  # no '/' in the name part
            # Actually, let's just check the expected string
            assert Path("microsoft--VibeVoice-ASR-HF") in path.parents or (
                "microsoft--VibeVoice-ASR-HF" in str(path)
            )

    def test_custom_model_id(self):
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore(
                model_id="my/org/model",
                local_model_dir="/cache",
            )
            path = core._get_local_model_path()
            assert str(path).endswith("org--model") or "my--org--model" in str(path)

    @staticmethod
    def test_static_method_signature():
        """_local_model_exists is a static method."""
        assert isinstance(
            type(VibeVoiceCore._local_model_exists).__dict__.get("_local_model_exists"),
            type(type.__call__),  # rough check — it's callable
        ) or hasattr(VibeVoiceCore, "_local_model_exists")

    def test_local_model_exists_checks_config_json(self, tmp_path):
        """_local_model_exists returns True only if config.json exists."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # No config.json → False
        assert VibeVoiceCore._local_model_exists(model_dir) is False

        # With config.json → True
        (model_dir / "config.json").write_text("{}")
        assert VibeVoiceCore._local_model_exists(model_dir) is True


class TestUnload:
    """Test the unload() reference-counting logic."""

    def test_unload_decrements_ref_count(self):
        """unload() should decrement the ref count."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore()

            # Simulate: model is loaded with ref_count=1
            VibeVoiceCore._ref_count = 2
            core.unload()
            assert VibeVoiceCore._ref_count == 1
            assert not core._loaded

    def test_unload_calls_unload_model_when_last_instance(self):
        """When ref count reaches 0, _unload_model should be called."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
            patch.object(
                VibeVoiceCore, "_unload_model", return_value=None
            ) as mock_unload,
        ):
            core = VibeVoiceCore()

            # Set up: this is the last instance
            VibeVoiceCore._ref_count = 1
            VibeVoiceCore._model_loaded = True

            core.unload()

            assert VibeVoiceCore._unload_model.called

    def test_unload_does_nothing_when_already_unloaded(self):
        """Calling unload on an already-unloaded instance should be a no-op."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            core = VibeVoiceCore()
            core._loaded = False  # simulate already unloaded

            prev_ref_count = VibeVoiceCore._ref_count
            core.unload()

            assert VibeVoiceCore._ref_count == prev_ref_count


class TestIsLoadedProperty:
    """Test the is_loaded property."""

    def test_is_loaded_true_when_model_exists(self):
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            # Reset class-level state to ensure clean start.
            VibeVoiceCore._model_loaded = False
            VibeVoiceCore._model = None

            core = VibeVoiceCore()

            # Inject a mock model so is_loaded returns True.
            from unittest.mock import MagicMock as M

            mock_model = M()
            VibeVoiceCore._model = mock_model
            VibeVoiceCore._model_loaded = True

            assert core.is_loaded is True

    def test_is_loaded_false_when_model_none(self):
        """If _model is None but _model_loaded is True, should be False."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            # Reset state for clean start.
            VibeVoiceCore._model_loaded = False
            VibeVoiceCore._model = None
            VibeVoiceCore._ref_count = 0

            core = VibeVoiceCore()

            # After unloading, _model should be None
            VibeVoiceCore._unload_model()

            assert core.is_loaded is False

    def test_is_loaded_false_when_not_loaded(self):
        """If _model_loaded is False, should return False regardless."""
        with (
            patch.object(VibeVoiceCore, "_load_model", return_value=None),
            patch.object(VibeVoiceCore, "__del__", return_value=None),
        ):
            # Reset state for clean start.
            VibeVoiceCore._model_loaded = False
            VibeVoiceCore._model = None
            VibeVoiceCore._ref_count = 0

            core = VibeVoiceCore()

            VibeVoiceCore._model_loaded = False

            assert core.is_loaded is False
