"""Baseline tests for the VibeVoice CLI argument parser and entry point."""

import pytest

from vibevoice import HOP_LENGTH, build_cli_parser


class TestCLIParser:
    """Tests for build_cli_parser()."""

    def test_parser_returns_argumentparser(self):
        parser = build_cli_parser()
        assert type(parser).__name__ == "ArgumentParser"

    def test_required_positional_input_path(self):
        """input_path is a required positional argument."""
        with pytest.raises(SystemExit):
            build_cli_parser().parse_args([])

    def test_required_positional_output_dir(self):
        """output_dir is a required positional argument."""
        with pytest.raises(SystemExit):
            build_cli_parser().parse_args(["input.wav"])

    def test_basic_parsing(self, tmp_path):
        parser = build_cli_parser()
        args = parser.parse_args(["audio/test.wav", str(tmp_path)])
        assert args.input_path == "audio/test.wav"
        assert str(args.output_dir) == str(tmp_path)

    def test_hotwords_option(self):
        parser = build_cli_parser()
        args = parser.parse_args(
            ["input.wav", "output/", "--hotwords", "VibeVoice,ASR"]
        )
        assert args.hotwords == "VibeVoice,ASR"

    def test_hotwords_default_none(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.hotwords is None

    def test_verbose_flag_long(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/", "--verbose"])
        assert args.verbose is True

    def test_verbose_flag_short(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/", "-v"])
        assert args.verbose is True

    def test_no_recursive_flag(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/", "--no-recursive"])
        assert args.no_recursive is True

    def test_no_recursive_default_false(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.no_recursive is False

    def test_model_option(self):
        parser = build_cli_parser()
        model_id = "my/custom-model"
        args = parser.parse_args(["input.wav", "output/", "--model", model_id])
        assert args.model == model_id

    def test_model_default(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.model == "microsoft/VibeVoice-ASR-HF"

    def test_device_option_cuda(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/", "--device", "cuda"])
        assert args.device == "cuda"

    def test_device_option_cpu(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/", "--device", "cpu"])
        assert args.device == "cpu"

    def test_device_invalid_value_rejected(self):
        """Invalid device value should cause a parse error."""
        with pytest.raises(SystemExit):
            build_cli_parser().parse_args(["input.wav", "output/", "--device", "tpu"])

    def test_device_default_none(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.device is None

    def test_acoustic_chunk_size_option(self):
        parser = build_cli_parser()
        args = parser.parse_args(
            ["input.wav", "output/", "--acoustic-chunk-size", "640000"]
        )
        assert args.acoustic_chunk_size == 640000

    def test_acoustic_chunk_size_default_none(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.acoustic_chunk_size is None

    def test_local_model_dir_option(self, tmp_path):
        parser = build_cli_parser()
        model_dir = str(tmp_path / "models")
        args = parser.parse_args(
            ["input.wav", "output/", "--local-model-dir", model_dir]
        )
        assert args.local_model_dir == model_dir

    def test_local_model_dir_default_none(self):
        parser = build_cli_parser()
        args = parser.parse_args(["input.wav", "output/"])
        assert args.local_model_dir is None


class TestCLIValidation:
    """Tests for the main() validation logic (chunk size must be multiple of hop length)."""

    def test_valid_chunk_size_accepted(self):
        """Chunk size that is a multiple of HOP_LENGTH should parse."""
        parser = build_cli_parser()
        args = parser.parse_args(
            ["input.wav", "output/", "--acoustic-chunk-size", str(HOP_LENGTH * 10)]
        )
        assert args.acoustic_chunk_size == HOP_LENGTH * 10

    def test_invalid_chunk_size_via_main_raises_system_exit(self):
        """main() should reject chunk sizes not divisible by HOP_LENGTH."""
        from vibevoice import main

        with pytest.raises(SystemExit):
            main(
                argv=[
                    "input.wav",
                    "output/",
                    "--acoustic-chunk-size",
                    str(HOP_LENGTH * 10 + 7),  # not a multiple of 3200
                ]
            )
