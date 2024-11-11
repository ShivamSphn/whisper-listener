import sys
import json
import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency imports with error handling
try:
    import torch
    import numpy as np
    import torchaudio
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available
    import requests
    from pyannote.audio import Pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read
except ImportError as e:
    logger.error(f"Failed to import required package: {str(e)}")
    logger.error("Please install all required packages from requirements.txt")
    sys.exit(1)

def check_system_compatibility() -> Dict[str, bool]:
    """Check system compatibility for various features."""
    compatibility = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "flash_attention": is_flash_attn_2_available(),
        "torchaudio": hasattr(torchaudio, "load"),
    }
    
    logger.info("System compatibility:")
    for feature, available in compatibility.items():
        logger.info(f"{feature}: {'✓' if available else '✗'}")
    
    return compatibility

def validate_device(device: str) -> str:
    """Validate and return appropriate device based on system capabilities."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        try:
            device_id = int(device.split(":")[-1])
            if device_id >= torch.cuda.device_count():
                logger.warning(f"CUDA device {device_id} not found, using device 0")
                return "cuda:0"
        except ValueError:
            return "cuda:0"
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
    return device

def preprocess_audio(inputs: Union[str, bytes, Dict[str, Any], np.ndarray]) -> tuple:
    """Preprocess audio input for diarization."""
    try:
        if isinstance(inputs, str):
            if inputs.startswith(("http://", "https://")):
                try:
                    response = requests.get(inputs, timeout=10)
                    response.raise_for_status()
                    inputs = response.content
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Failed to download audio from URL: {str(e)}")
            else:
                if not Path(inputs).exists():
                    raise FileNotFoundError(f"Audio file not found: {inputs}")
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            try:
                inputs = ffmpeg_read(inputs, 16000)
            except Exception as e:
                raise ValueError(f"Failed to read audio file: {str(e)}")

        if isinstance(inputs, dict):
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "Dictionary input must contain 'raw'/'array' and 'sampling_rate' keys"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != 16000:
                inputs = torchaudio.functional.resample(
                    torch.from_numpy(inputs), in_sampling_rate, 16000
                ).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"Expected numpy ndarray, got {type(inputs)}")
        if len(inputs.shape) != 1:
            raise ValueError("Expected single channel audio input")

        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        raise

def diarize_audio(diarizer_inputs, diarization_pipeline, num_speakers=None, min_speakers=None, max_speakers=None):
    """Perform speaker diarization on audio input."""
    try:
        diarization = diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": 16000},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            })

        if not segments:
            logger.warning("No speaker segments found in the audio")
            return []

        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                new_segments.append({
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                })
                prev_segment = segments[i]

        new_segments.append({
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        })

        return new_segments

    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        raise

def post_process_segments(segments, transcript, group_by_speaker=False):
    """Align diarization segments with transcription."""
    try:
        if not transcript:
            logger.warning("Empty transcript received")
            return []

        end_timestamps = np.array([
            chunk["timestamp"][-1] if chunk["timestamp"][-1] is not None 
            else sys.float_info.max for chunk in transcript
        ])
        segmented_preds = []

        for segment in segments:
            end_time = segment["segment"]["end"]
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))

            if group_by_speaker:
                segmented_preds.append({
                    "speaker": segment["speaker"],
                    "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                })
            else:
                for i in range(upto_idx + 1):
                    segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

            transcript = transcript[upto_idx + 1:]
            end_timestamps = end_timestamps[upto_idx + 1:]

            if len(end_timestamps) == 0:
                break

        return segmented_preds

    except Exception as e:
        logger.error(f"Post-processing failed: {str(e)}")
        raise

def transcribe_with_diarization(
    audio_path: str,
    output_path: str = "output.json",
    model_name: str = "openai/whisper-large-v3",
    diarization_model: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 24,
    chunk_length_s: int = 30,
    task: str = "transcribe",
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_flash: bool = False,
    return_timestamps: str = "chunk",
) -> Dict[str, Any]:
    """
    Transcribe audio with speaker diarization.
    
    Args:
        audio_path: Path to audio file
        output_path: Path to save JSON output
        model_name: Whisper model to use
        diarization_model: Diarization model to use
        hf_token: HuggingFace token for diarization model
        device: Device to run on ("cuda:0", "cuda:1", "mps", etc.)
        batch_size: Batch size for transcription
        chunk_length_s: Length of audio chunks in seconds
        task: "transcribe" or "translate"
        language: Language code (None for auto-detection)
        num_speakers: Exact number of speakers (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        use_flash: Whether to use Flash Attention 2
        return_timestamps: "chunk" or "word" level timestamps
    """
    try:
        # Check system compatibility
        compatibility = check_system_compatibility()
        
        # Validate device
        device = validate_device(device)
        
        # Configure Flash Attention
        model_kwargs = {}
        if use_flash:
            if not compatibility["flash_attention"]:
                logger.warning("Flash Attention 2 not available, falling back to SDPA")
                model_kwargs = {
                    "attn_implementation": "sdpa",
                    "use_flash_attention_2": False
                }
            else:
                model_kwargs = {
                    "attn_implementation": "flash_attention_2",
                    "use_flash_attention_2": True
                }
        else:
            model_kwargs = {
                "attn_implementation": "sdpa",
                "use_flash_attention_2": False
            }

        # Initialize ASR pipeline
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device=device,
                model_kwargs=model_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ASR pipeline: {str(e)}")

        # Initialize diarization pipeline if token provided
        if hf_token:
            try:
                diarization_pipeline = Pipeline.from_pretrained(
                    checkpoint_path=diarization_model,
                    use_auth_token=hf_token,
                )
                diarization_pipeline.to(torch.device(device))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize diarization pipeline: {str(e)}")

        # Prepare audio input
        inputs, diarizer_inputs = preprocess_audio(audio_path)

        # Configure transcription parameters
        generate_kwargs = {"task": task}
        if language:
            generate_kwargs["language"] = language
        if pipe.model.config.model_type.split(".")[-1] == "en":
            generate_kwargs.pop("task")

        ts = True if return_timestamps == "chunk" else "word"

        # Perform transcription
        try:
            outputs = pipe(
                audio_path,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=ts,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")

        # Perform diarization if token provided
        if hf_token:
            segments = diarize_audio(
                diarizer_inputs, 
                diarization_pipeline,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            result = post_process_segments(segments, outputs["chunks"], group_by_speaker=False)
        else:
            result = outputs

        # Save results
        try:
            with open(output_path, "w", encoding="utf8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {str(e)}")
            # Continue execution as this is not a critical error

        return result

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    result = transcribe_with_diarization(
        audio_path="path/to/audio.mp3",
        hf_token="your_huggingface_token",  # Required for diarization
        device="cuda:0",  # or "mps" for Mac
        num_speakers=2,   # Optional: specify exact number of speakers
        # min_speakers=2, # Or specify min/max range
        # max_speakers=4,
        use_flash=True    # Use Flash Attention 2 if available
    )
