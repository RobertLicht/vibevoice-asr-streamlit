from pathlib import Path

import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# Modell card on huggingface: https://huggingface.co/microsoft/VibeVoice-ASR-HF

# Check availability of GPU
print(f"Utilized version of torch: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected hardware accelerator: {device}")
if device == "cuda":
    print(f"Utilized hardware: {torch.cuda.get_device_name(0)}")

# 1. Specify the model ID
model_id = "microsoft/VibeVoice-ASR-HF"

# 1.1 Local model directory (optional)
# If set, the model will be loaded from this directory first.
# It will be downloaded from Hugging Face only if not found locally.
local_model_dir = "models"  # or None to use standard HF download

# Compute the local model path and check if model exists locally
_model_dir_name = model_id.replace("/", "--")
_local_model_path = Path(local_model_dir) / _model_dir_name if local_model_dir else None
_local_files_only = False
_load_path = model_id
if _local_model_path is not None and (_local_model_path / "config.json").is_file():
    _local_files_only = True
    _load_path = str(_local_model_path)
    print(f"Loading model from local path: {_local_model_path}")
else:
    print(f"Loading model from Hugging Face (model_id: {model_id})")

# 1.2 Select a tokenizer chunk size
#    Must be a multiple of the hop length (3200 for the original acoustic tokenizer)
tokenizer_chunk_size = 640_000  # default is 1_440_000 (60s @ 24kHz)

# 2. Load the processor and model
print("Loading model (this might take a few minutes on the first run)...")
processor = AutoProcessor.from_pretrained(
    _load_path, local_files_only=_local_files_only
)

# Using device_map="auto" will automatically place the model on a GPU if you have one.
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
    _load_path,
    local_files_only=_local_files_only,
    device_map=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

print(f"\nModel loaded on {model.device} with dtype {model.dtype}\n")

# 3. Define the audio path and optional hotwords context
# audio_path = "./Anamnese_TVT_Rio_Neumann_20260519.wav"  # Replace with your actual audio file path
audio_path = "./data/audio/URecorder_Grillen.wav"
hotwords = ""  # Optional: provide custom terms, names, or jargon here

# 4. Prepare the transcription request
print(f"Transcribing {audio_path}...\n")
inputs = processor.apply_transcription_request(
    audio=audio_path,
    prompt=hotwords,
).to(model.device, model.dtype)

# 5. Generate text
with torch.no_grad():
    output_ids = model.generate(**inputs, tokenizer_chunk_size=tokenizer_chunk_size)
print("Generation complete!\n")

# 6. Decode the results
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
""""
transcription = processor.batch_decode(
    generated_ids, skip_special_tokens=True, return_format="transcription_only"
)[0]

print("\n--- Transcription ---")
print(transcription)
"""

transcription = processor.decode(generated_ids, return_format="parsed")[0]
print("\n" + "=" * 60)
print("TRANSCRIPTION (list of dicts)")
print("=" * 60)
for speaker_transcription in transcription:
    print(speaker_transcription)
