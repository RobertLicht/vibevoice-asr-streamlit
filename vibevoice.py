"""
VibeVoice Core — Transcription pipeline for the VibeVoice-ASR model.

Provides a VibeVoiceCore class that loads the model once and transcribes
one or many audio files, saving structured JSON output (return_format="parsed").
"""

import argparse
import gc
import json
import logging
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

# Suppress the deprecated `feature_extractor_class` warning from
# VibeVoiceAsrProcessor (upstream transformers issue — see
# https://github.com/huggingface/transformers). The model still
# works correctly; this is purely a style-of-registration warning.
warnings.filterwarnings(
    "ignore",
    message=".*feature_extractor_class.*is deprecated.*",
    category=DeprecationWarning,
)


# Accepted audio file extensions
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}

# Tokenizer hop length (must be a multiple of this)
HOP_LENGTH = 3200

# Default acoustic tokenizer chunk size (60 seconds @ 24 kHz)
DEFAULT_CHUNK_SIZE = 1_440_000

# Note: the parameter passed to model.generate() is
# "acoustic_tokenizer_chunk_size" (per the official Transformers docs).
# Earlier model card examples used the shorter "tokenizer_chunk_size" alias,
# which was dropped in a later transformers release.

# Audio sample rate expected by the tokenizer
SAMPLE_RATE = 24_000

# Resample modes
RESAMPLE_KEEP_ORIGINAL = "keep_original"
RESAMPLE_DOWN_TO_24K = "down_to_24k"
RESAMPLE_DOWN_TO_32K = "down_to_32k"
RESAMPLE_DOWN_TO_16K = "down_to_16k"
RESAMPLE_MODES = (
    RESAMPLE_KEEP_ORIGINAL,
    RESAMPLE_DOWN_TO_24K,
    RESAMPLE_DOWN_TO_32K,
    RESAMPLE_DOWN_TO_16K,
)


class VibeVoiceCore:
    """Core wrapper around the VibeVoice-ASR model.

    Loads the model once and provides methods to transcribe audio files
    and directories, saving structured JSON output.

    The model and processor are stored as class-level attributes so that all
    instances share a single copy.  A reference counter tracks how many
    instances are alive — the model is only unloaded when the last instance
    is destroyed (e.g. on page reload or config change).  This prevents the
    model from stacking in memory when the user refreshes the page or creates
    multiple instances.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (default: "microsoft/VibeVoice-ASR-HF").
    device : str, optional
        Device to run inference on. Auto-detected if None.
    acoustic_chunk_size : int, optional
        Acoustic tokenizer chunk size (must be a multiple of 3200).
        Defaults to 1,440,000 (60 s @ 24 kHz).
    verbose : bool
        If True, enable INFO-level logging.
    local_model_dir : str or None, optional
        Base directory (relative or absolute) where models are stored
        locally.  If set, the model is loaded from this directory first;
        it is downloaded from Hugging Face only if not found locally.
        The full path for a model is
        ``{local_model_dir}/{model_id with '/' replaced by '--'}/``.
        Defaults to ``None`` (standard Hugging Face download behaviour).
    """

    # -- Class-level singleton state -----------------------------------
    _model: Optional["VibeVoiceAsrForConditionalGeneration"] = None
    _processor = None  # AutoProcessor
    _model_loaded: bool = False
    _ref_count: int = 0  # number of live instances

    def __init__(
        self,
        model_id: str = "microsoft/VibeVoice-ASR-HF",
        device: Optional[str] = None,
        acoustic_chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = False,
        local_model_dir: Optional[str] = None,
    ):
        # Validate chunk size
        if acoustic_chunk_size % HOP_LENGTH != 0:
            raise ValueError(
                f"acoustic_chunk_size ({acoustic_chunk_size}) must be a multiple "
                f"of hop length ({HOP_LENGTH})."
            )

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.acoustic_chunk_size = acoustic_chunk_size
        self.verbose = verbose
        self.local_model_dir = local_model_dir

        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # --- Load or share the class-level model ---------------------
        # Signal that loading is (or will be) happening so that __del__ of
        # stale instances does NOT unload the model while we are in the
        # middle of initialisation.
        VibeVoiceCore._model_loaded = True

        # Garbage-collect stale instances before we potentially load a new
        # model, so their __del__ runs early and updates the reference counter.
        gc.collect()

        self._load_model()

        # This instance is alive and using the model.
        self._loaded = True

    def __del__(self):
        """Decrement the reference counter; unload if this was the last instance."""
        if not self._loaded:
            return
        if VibeVoiceCore._ref_count > 0:
            VibeVoiceCore._ref_count -= 1
        # Only unload if model is still loaded (i.e. no new instance has
        # taken over) AND there are no other live instances.
        if VibeVoiceCore._model_loaded and VibeVoiceCore._ref_count <= 0:
            VibeVoiceCore._model_loaded = False
            VibeVoiceCore._ref_count = 0
            self._unload_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_local_model_path(self) -> Optional[Path]:
        """Return the full local model directory for this model, or ``None``.

        The path is derived as
        ``{local_model_dir}/{model_id with '/' replaced by '--'}/``.
        """
        if self.local_model_dir is None:
            return None
        model_dir_name = self.model_id.replace("/", "--")
        return Path(self.local_model_dir) / model_dir_name

    @staticmethod
    def _local_model_exists(local_path: Path) -> bool:
        """Return True if the model directory contains at least ``config.json``."""
        return (local_path / "config.json").is_file()

    def _load_from_local(self, local_path: Path):
        """Load processor and model from a local directory."""
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        logger.info("Loading model from local path: %s", local_path)
        VibeVoiceCore._processor = AutoProcessor.from_pretrained(
            str(local_path), local_files_only=True
        )
        VibeVoiceCore._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            str(local_path),
            local_files_only=True,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        VibeVoiceCore._model_loaded = True

        self.processor = VibeVoiceCore._processor
        self.model = VibeVoiceCore._model
        VibeVoiceCore._ref_count = 1
        logger.info(
            "Model loaded from local on %s with dtype %s",
            self.model.device,
            self.model.dtype,
        )

    @staticmethod
    def _download_to_local(model_id: str, local_model_dir: str) -> Path:
        """Download the model from Hugging Face into a local directory.

        Returns the path to the downloaded model.
        """
        from huggingface_hub import snapshot_download

        model_dir_name = model_id.replace("/", "--")
        local_path = Path(local_model_dir) / model_dir_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
        )
        return local_path

    def _load_model(self):
        """Load processor and model using a local-first strategy.

        Tier 1 — Load from local directory (if ``local_model_dir`` is set
        and the model exists there).

        Tier 2 — Download from Hugging Face into the local directory,
        then load from local (if ``local_model_dir`` is set but the
        local model is missing or corrupted).

        Tier 3 — Standard Hugging Face loading (fallback when local
        directory is not configured or download fails).
        """
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        # -- Reuse existing class-level model if already loaded ----------
        if VibeVoiceCore._model_loaded and VibeVoiceCore._model is not None:
            self.processor = VibeVoiceCore._processor
            self.model = VibeVoiceCore._model
            VibeVoiceCore._ref_count += 1
            logger.info(
                "Reusing existing model (ref_count=%d, device=%s, dtype=%s)",
                VibeVoiceCore._ref_count,
                self.model.device,
                self.model.dtype,
            )
            return

        local_path = self._get_local_model_path()

        # -- Tier 1: Load from local ------------------------------------
        if local_path is not None and self._local_model_exists(local_path):
            try:
                self._load_from_local(local_path)
                return
            except Exception as e:
                logger.warning(
                    "Failed to load model from local (%s) — falling back: %s",
                    local_path,
                    e,
                )

        # -- Tier 2: Download from HF to local, then load ----------------
        if local_path is not None:
            assert self.local_model_dir is not None  # guaranteed by local_path
            try:
                logger.info("Downloading model from Hugging Face to %s ...", local_path)
                self._download_to_local(self.model_id, self.local_model_dir)
                self._load_from_local(local_path)
                return
            except Exception as e:
                logger.warning(
                    "Failed to download to local: %s — falling back to HF.", e
                )

        # -- Tier 3: Standard Hugging Face loading (fallback) ------------
        logger.info("Loading model from Hugging Face (%s) ...", self.model_id)
        VibeVoiceCore._processor = AutoProcessor.from_pretrained(self.model_id)
        VibeVoiceCore._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        VibeVoiceCore._model_loaded = True

        self.processor = VibeVoiceCore._processor
        self.model = VibeVoiceCore._model
        VibeVoiceCore._ref_count = 1
        logger.info(
            "Model loaded on %s with dtype %s",
            self.model.device,
            self.model.dtype,
        )

    @staticmethod
    def _unload_model():
        """Delete class-level references to the model and free memory."""
        logger.info("Unloading model from memory ...")
        if VibeVoiceCore._model is not None:
            del VibeVoiceCore._model
        if VibeVoiceCore._processor is not None:
            del VibeVoiceCore._processor
        VibeVoiceCore._model = None
        VibeVoiceCore._processor = None
        VibeVoiceCore._model_loaded = False
        VibeVoiceCore._ref_count = 0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def unload(self):
        """Unload model and free GPU/CPU memory.

        Decrements the reference counter. The model is only freed when the
        counter reaches zero (i.e. no other live instances depend on it).
        This should be called before discarding the VibeVoiceCore instance,
        e.g. when the user changes the chunk size or reloads the web page.
        """
        if not self._loaded:
            return  # already unloaded

        # Decrement and clear — this instance no longer uses the model.
        if VibeVoiceCore._ref_count > 0:
            VibeVoiceCore._ref_count -= 1
        self._loaded = False
        self.model = None
        self.processor = None

        # If no other instances need the model, unload it entirely.
        if VibeVoiceCore._model_loaded and VibeVoiceCore._ref_count <= 0:
            VibeVoiceCore._model_loaded = False
            VibeVoiceCore._ref_count = 0
            VibeVoiceCore._unload_model()

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory.

        This checks the class-level state so that all instances see the
        same loading status.
        """
        return VibeVoiceCore._model_loaded and VibeVoiceCore._model is not None

    @staticmethod
    def _select_chunk_size(
        audio_samples: int,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> int:
        """Select an optimal tokenizer chunk size based on audio length.

        The model is designed for up to 60 minutes of audio in a single pass.
        The default chunk size of 1,440,000 (60 s) is used for most cases.
        For very short audio, a smaller chunk reduces memory overhead.

        Parameters
        ----------
        audio_samples : int
            Number of samples in the audio (at 24 kHz).
        max_chunk_size : int
            Upper bound for chunk size (default: 1,440,000).

        Returns
        -------
        int
            Selected chunk size (multiple of 3200).
        """
        # Duration in seconds at 24 kHz
        duration_seconds = audio_samples / SAMPLE_RATE

        # For audio shorter than 10 minutes, scale chunk down proportionally
        # but keep it a reasonable minimum to avoid excessive chunk boundaries
        if duration_seconds <= 600:  # ≤ 10 minutes
            # Target chunk ≈ 10 s for very short, up to 60 s for 10 min audio
            target_chunk_seconds = max(10, min(duration_seconds, 60))
            chunk_samples = int(target_chunk_seconds * SAMPLE_RATE)
        else:
            chunk_samples = max_chunk_size

        # Snap to nearest multiple of hop length
        chunk_samples = int(round(chunk_samples / HOP_LENGTH) * HOP_LENGTH)

        # Ensure at least one hop length
        chunk_samples = max(HOP_LENGTH, chunk_samples)

        logger.debug(
            "Selected chunk_size=%d (audio: %d samples, %.1f s)",
            chunk_samples,
            audio_samples,
            duration_seconds,
        )
        return chunk_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_audio_if_needed(
        audio_path: Path,
        resample_mode: str,
    ) -> Tuple[Path, bool]:
        """Load, optionally resample, and save audio to a temp file.

        *resample_mode* controls the behaviour:

        - **keep_original** — never resample; return the original path.
        - **down_to_24k** — resample to 24 kHz only if the file's sample
          rate exceeds 24 kHz.
        - **down_to_32k** — resample to 32 kHz only if the file's sample
          rate exceeds 32 kHz.
        - **down_to_16k** — resample to 16 kHz only if the file's sample
          rate exceeds 16 kHz.

        If the file's native sample rate is at or below the target, the
        original path is returned unchanged (``is_temp=False``).

        Parameters
        ----------
        audio_path : Path
            Path to the source audio file.
        resample_mode : str
            One of ``RESAMPLE_KEEP_ORIGINAL``, ``RESAMPLE_DOWN_TO_24K``,
            ``RESAMPLE_DOWN_TO_32K``, or ``RESAMPLE_DOWN_TO_16K``.

        Returns
        -------
        Tuple[Path, bool]
            (effective_path, is_temp) — the file to pass to the model and a
            flag indicating whether the file was created temporarily.
        """
        if resample_mode == RESAMPLE_KEEP_ORIGINAL:
            return audio_path, False

        waveform, orig_sr = torchaudio.load(str(audio_path))

        # Determine the target sample rate from the mode
        if resample_mode == RESAMPLE_DOWN_TO_24K:
            target_sr = 24_000
        elif resample_mode == RESAMPLE_DOWN_TO_32K:
            target_sr = 32_000
        elif resample_mode == RESAMPLE_DOWN_TO_16K:
            target_sr = 16_000
        else:
            # Unknown mode — skip resampling as a safety net
            logger.warning(
                "Unknown resample_mode '%s' — skipping resample for %s",
                resample_mode,
                audio_path.name,
            )
            return audio_path, False

        if orig_sr <= target_sr:
            # Already at or below the target — no work to do.
            logger.debug(
                "Audio %s is %d Hz (mode=%s, target=%d) — skipping resample.",
                audio_path.name,
                orig_sr,
                resample_mode,
                target_sr,
            )
            return audio_path, False

        logger.info(
            "Resampling %s from %d Hz → %d Hz (mode=%s) ...",
            audio_path.name,
            orig_sr,
            target_sr,
            resample_mode,
        )

        resample = T.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
        )
        resampled = resample(waveform)

        # Write to a temporary file with the same suffix
        suffix = audio_path.suffix or ".wav"
        fd, tmp_path_str = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # close the fd so we can write via torchaudio
        tmp_path = Path(tmp_path_str)

        torchaudio.save(str(tmp_path), resampled, target_sr)
        logger.info("Saved resampled audio to %s", tmp_path)
        return tmp_path, True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Union[str, Path],
        hotwords: Optional[str] = None,
        acoustic_chunk_size: Optional[int] = None,
        resample_mode: str = RESAMPLE_KEEP_ORIGINAL,
    ) -> List[Dict]:
        """Transcribe a single audio file and return the parsed output.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio file.
        hotwords : str or None
            Optional hotwords to guide transcription.
        acoustic_chunk_size : int or None
            Override the acoustic tokenizer chunk size for this transcription.
        resample_mode : str
            Controls audio resampling before transcription. One of
            ``RESAMPLE_KEEP_ORIGINAL`` (default), ``RESAMPLE_DOWN_TO_24K``,
            or ``RESAMPLE_DOWN_TO_16K``.  Downsampling reduces data volume
            and processing time for high-sample-rate audio.

        Returns
        -------
        list[dict]
            Parsed transcription (list of dicts with speaker/timestamp/content).
        """
        audio_path = Path(audio_path)
        if not self.is_loaded:
            raise RuntimeError(
                "Model has been unloaded. Create a new VibeVoiceCore instance "
                "or call _load_model() to reload."
            )
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cs = acoustic_chunk_size or self.acoustic_chunk_size

        logger.info("Transcribing %s (chunk_size=%d) ...", audio_path.name, cs)
        t0 = time.monotonic()

        # --- Optional resampling ----------------------------------------
        effective_path: Path = audio_path
        temp_path: Optional[Path] = None

        if resample_mode != RESAMPLE_KEEP_ORIGINAL:
            effective_path, is_temp = self._resample_audio_if_needed(
                audio_path, resample_mode
            )
            if is_temp:
                temp_path = effective_path

        try:
            inputs = self.processor.apply_transcription_request(
                audio=str(effective_path),
                prompt=hotwords if hotwords else "",
            ).to(self.model.device, self.model.dtype)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, acoustic_tokenizer_chunk_size=cs
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            transcription = self.processor.decode(
                generated_ids, return_format="parsed"
            )[0]

        finally:
            # Clean up any temporary resampled file
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
                logger.debug("Removed temp resampled file %s", temp_path)

        elapsed = time.monotonic() - t0
        logger.info(
            "Transcribed %s in %.1f s (%d utterances)",
            audio_path.name,
            elapsed,
            len(transcription),
        )

        return transcription

    def transcribe_and_save(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        hotwords: Optional[str] = None,
        acoustic_chunk_size: Optional[int] = None,
        resample_mode: str = RESAMPLE_KEEP_ORIGINAL,
    ) -> Path:
        """Transcribe a single audio file and save JSON output.

        The output JSON contains a ``source_file`` field and a
        ``transcription`` field (list of dicts from ``return_format="parsed"``).

        Parameters
        ----------
        audio_path : str or Path
        output_dir : str or Path
        hotwords : str or None
        acoustic_chunk_size : int or None
        resample_mode : str
            Controls audio resampling before transcription.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        transcription = self.transcribe(
            audio_path,
            hotwords=hotwords,
            acoustic_chunk_size=acoustic_chunk_size,
            resample_mode=resample_mode,
        )

        output_file = output_dir / f"{audio_path.stem}.json"
        payload = {
            "source_file": str(audio_path.resolve()),
            "transcription": transcription,
        }

        output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved: %s", output_file)
        return output_file

    @staticmethod
    def discover_audio_files(
        input_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Path]:
        """Discover audio files under *input_path*.

        If *input_path* is a file, returns a single-element list (after
        validating the extension). If *input_path* is a directory,
        returns all files with recognized audio extensions.

        Parameters
        ----------
        input_path : str or Path
        recursive : bool
            If True, search subdirectories (default: True).

        Returns
        -------
        list[Path]
        """
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(
                    f"Unknown audio extension: '{input_path.suffix}' for {input_path}"
                )
            return [input_path]

        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        pattern = "**/*" if recursive else "*"
        files = sorted(input_path.glob(pattern))
        audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]

        logger.info("Discovered %d audio file(s) in %s", len(audio_files), input_path)
        return audio_files

    def transcribe_batch(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        hotwords: Optional[str] = None,
        acoustic_chunk_size: Optional[int] = None,
        recursive: bool = True,
        resample_mode: str = RESAMPLE_KEEP_ORIGINAL,
    ) -> List[Path]:
        """Transcribe all audio files under *input_path* and save JSONs.

        Parameters
        ----------
        input_path : str or Path
            Path to an audio file or a directory of audio files.
        output_dir : str or Path
            Directory where JSON transcriptions will be saved.
        hotwords : str or None
            Optional hotwords to guide transcription for all files.
        acoustic_chunk_size : int or None
            Override the acoustic tokenizer chunk size for all files.
        recursive : bool
            Search subdirectories when input_path is a directory.
        resample_mode : str
            Controls audio resampling before transcription for all files.

        Returns
        -------
        list[Path]
            Paths to the saved JSON files, in processing order.
        """
        audio_files = self.discover_audio_files(input_path, recursive=recursive)

        if not audio_files:
            logger.warning("No audio files found at %s", input_path)
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files: List[Path] = []

        batch_start = time.monotonic()

        for i, audio_file in enumerate(audio_files, 1):
            logger.info("Processing %d/%d: %s", i, len(audio_files), audio_file.name)
            try:
                out = self.transcribe_and_save(
                    audio_file,
                    output_dir,
                    hotwords=hotwords,
                    acoustic_chunk_size=acoustic_chunk_size,
                    resample_mode=resample_mode,
                )
                output_files.append(out)
            except Exception as e:
                logger.error("Failed to transcribe %s: %s", audio_file, e)

        elapsed = time.monotonic() - batch_start
        logger.info(
            "Batch complete: %d/%d file(s) transcribed in %.1f s",
            len(output_files),
            len(audio_files),
            elapsed,
        )

        return output_files


# ======================================================================
# CLI Entry Point
# ======================================================================


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vibevoice",
        description="VibeVoice ASR — Transcribe audio files to structured JSON.",
    )

    parser.add_argument(
        "input_path",
        help="Path to an audio file or a directory containing audio files.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where JSON transcriptions will be saved.",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="Optional hotwords / context to guide transcription.",
    )
    parser.add_argument(
        "--acoustic-chunk-size",
        type=int,
        default=None,
        help=(
            "Acoustic tokenizer chunk size (multiple of 3200, default 1440000). "
            "Only change if GPU memory is insufficient."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories when input is a directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (INFO-level) logging.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/VibeVoice-ASR-HF",
        help="Hugging Face model ID (default: microsoft/VibeVoice-ASR-HF).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device for inference (auto-detected by default).",
    )
    parser.add_argument(
        "--local-model-dir",
        default=None,
        help=(
            "Base directory where models are stored locally. The model is "
            "loaded from this directory first; it is downloaded from "
            "Hugging Face only if not found locally. E.g. 'models/' will "
            "store at models/microsoft--VibeVoice-ASR-HF/."
        ),
    )

    return parser


def main(argv=None):
    """CLI entry point."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # Validate chunk size if provided
    if (
        args.acoustic_chunk_size is not None
        and args.acoustic_chunk_size % HOP_LENGTH != 0
    ):
        parser.error(
            f"--acoustic-chunk-size must be a multiple of {HOP_LENGTH} "
            f"(got {args.acoustic_chunk_size})."
        )

    core = VibeVoiceCore(
        model_id=args.model,
        device=args.device,
        acoustic_chunk_size=args.acoustic_chunk_size or DEFAULT_CHUNK_SIZE,
        verbose=args.verbose,
        local_model_dir=args.local_model_dir,
    )

    recursive = not args.no_recursive
    core.transcribe_batch(
        input_path=args.input_path,
        output_dir=args.output_dir,
        hotwords=args.hotwords,
        acoustic_chunk_size=args.acoustic_chunk_size,
        recursive=recursive,
    )


if __name__ == "__main__":
    main()
