"""
VibeVoice Integration for realtime TTS.

Purpose: Generate speech from text using Microsoft's VibeVoice-Realtime-0.5B
Inputs: Text to synthesize
Outputs: Audio data/URL

Model: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- ~300ms latency for first audio chunk
- Streaming text input support
- Based on Qwen2.5-0.5B LLM
- Uses acoustic tokenizer with 7.5 Hz frame rate
"""

from typing import Any, Dict, Optional, AsyncIterator, List, Union
import asyncio
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Available speakers for VibeVoice-Realtime
AVAILABLE_SPEAKERS = [
    "Carter",    # Default English speaker
    "Aria",      # English
    "Guy",       # English
    "Jenny",     # English
    "Sara",      # English
]


class VibeVoiceRealtimeClient:
    """
    Client for VibeVoice-Realtime-0.5B TTS.
    
    Uses transformers library to load from HuggingFace cache.
    Model downloads on first use (~1GB).
    
    VibeVoice features:
    - ~300ms latency for first audio chunk
    - Streaming text input support
    - Based on Qwen2.5-0.5B LLM
    - Uses acoustic tokenizer with 7.5 Hz frame rate (ultra-low)
    """
    
    MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
    
    def __init__(
        self,
        speaker: str = "Carter",
        sample_rate: int = 24000,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize VibeVoice Realtime client.
        
        Args:
            speaker: Speaker name (Carter, Aria, Guy, Jenny, Sara)
            sample_rate: Output audio sample rate (default 24kHz)
            device: Device to run on (cuda, mps, cpu). Auto-detected if None.
            cache_dir: HuggingFace cache directory. Uses default if None.
        """
        self.speaker = speaker if speaker in AVAILABLE_SPEAKERS else "Carter"
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._is_loaded = False
        
        # Auto-detect device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
        
        logger.info(f"VibeVoice client initialized: speaker={self.speaker}, device={self.device}")
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def load_model(self) -> bool:
        """
        Load the VibeVoice-Realtime model from HuggingFace.
        
        Downloads to HuggingFace cache on first run (~1GB).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._is_loaded:
            return True
        
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            
            logger.info(f"Loading VibeVoice-Realtime from {self.MODEL_ID}...")
            logger.info("(First run will download ~1GB to HuggingFace cache)")
            
            # Load model and processor
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            
            self._model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                cache_dir=self.cache_dir,
            ).to(self.device)
            
            self._model.eval()
            self._is_loaded = True
            logger.info("VibeVoice-Realtime model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load VibeVoice model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def synthesize(self, text: str, speaker: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker: Optional speaker override
            
        Returns:
            Dictionary with:
                - audio: numpy array of audio samples
                - sample_rate: int sample rate
                - duration_ms: float duration in milliseconds
                - model: str model name
                - speaker: str speaker used
        """
        if not self.load_model():
            return {
                "audio": None,
                "error": "Model not loaded",
                "sample_rate": self.sample_rate,
            }
        
        speaker = speaker if speaker in AVAILABLE_SPEAKERS else self.speaker
        
        try:
            import torch
            import numpy as np
            
            # Prepare input
            inputs = self._processor(
                text=text,
                speaker=speaker,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate audio
            with torch.no_grad():
                outputs = self._model.generate(**inputs)
            
            # Convert to numpy
            audio_array = outputs.cpu().numpy().squeeze()
            duration_ms = (len(audio_array) / self.sample_rate) * 1000
            
            return {
                "audio": audio_array,
                "sample_rate": self.sample_rate,
                "duration_ms": duration_ms,
                "model": self.MODEL_ID,
                "speaker": speaker,
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "audio": None,
                "error": str(e),
                "sample_rate": self.sample_rate,
            }
    
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        speaker: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream synthesis for real-time applications.
        
        Generates audio chunks as text arrives, achieving ~300ms
        first-audio latency.
        
        Args:
            text_stream: Async iterator of text chunks
            speaker: Optional speaker override
            
        Yields:
            Audio chunk dictionaries with:
                - audio: numpy array chunk
                - sample_rate: int
                - is_final: bool
        """
        if not self.load_model():
            yield {
                "audio": None,
                "error": "Model not loaded",
                "is_final": True,
            }
            return
        
        speaker = speaker if speaker in AVAILABLE_SPEAKERS else self.speaker
        buffer = ""
        
        try:
            import torch
            
            async for text_chunk in text_stream:
                buffer += text_chunk
                
                # Process when we have enough text (sentence boundary or buffer limit)
                if self._should_process_chunk(buffer):
                    chunk_to_process = buffer
                    buffer = ""
                    
                    inputs = self._processor(
                        text=chunk_to_process,
                        speaker=speaker,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self._model.generate(**inputs)
                    
                    audio_chunk = outputs.cpu().numpy().squeeze()
                    
                    yield {
                        "audio": audio_chunk,
                        "sample_rate": self.sample_rate,
                        "is_final": False,
                    }
            
            # Process remaining buffer
            if buffer.strip():
                inputs = self._processor(
                    text=buffer,
                    speaker=speaker,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self._model.generate(**inputs)
                
                audio_chunk = outputs.cpu().numpy().squeeze()
                
                yield {
                    "audio": audio_chunk,
                    "sample_rate": self.sample_rate,
                    "is_final": True,
                }
                
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            yield {
                "audio": None,
                "error": str(e),
                "is_final": True,
            }
    
    def _should_process_chunk(self, buffer: str) -> bool:
        """Determine if buffer should be processed (sentence boundary or length)."""
        # Process at sentence boundaries or when buffer is long enough
        sentence_ends = [".", "!", "?", "\n"]
        if any(buffer.rstrip().endswith(end) for end in sentence_ends):
            return len(buffer) >= 10  # Min 10 chars
        return len(buffer) >= 200  # Max buffer before forced process
    
    def set_speaker(self, speaker: str) -> None:
        """Update the speaker voice."""
        if speaker in AVAILABLE_SPEAKERS:
            self.speaker = speaker
            logger.info(f"Speaker set to: {speaker}")
        else:
            logger.warning(f"Unknown speaker '{speaker}', available: {AVAILABLE_SPEAKERS}")
    
    @staticmethod
    def get_available_speakers() -> List[str]:
        """Get list of available speaker voices."""
        return AVAILABLE_SPEAKERS.copy()
    
    def save_audio(
        self,
        audio_data,
        output_path: str,
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        Save audio data to file.
        
        Args:
            audio_data: numpy array of audio samples
            output_path: Path to save audio file
            sample_rate: Optional sample rate override
            
        Returns:
            True if saved successfully
        """
        try:
            import scipy.io.wavfile as wav
            import numpy as np
            
            sr = sample_rate or self.sample_rate
            
            # Ensure proper dtype for wav
            if audio_data.dtype != np.int16:
                # Normalize and convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)
            
            wav.write(output_path, sr, audio_data)
            logger.info(f"Audio saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False


# Convenience alias
VibeVoiceClient = VibeVoiceRealtimeClient


class VibeVoiceSTT:
    """
    Speech-to-Text using OpenAI Whisper for realtime transcription.
    
    Complements VibeVoice TTS to provide full voice conversation support.
    Uses whisper-large-v3-turbo for fast, accurate transcription.
    """
    
    MODEL_ID = "openai/whisper-large-v3-turbo"
    
    def __init__(
        self,
        model_size: str = "turbo",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Whisper STT client.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, turbo)
            device: Device to run on (cuda, mps, cpu). Auto-detected if None.
            cache_dir: HuggingFace cache directory.
        """
        self.model_size = model_size
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._is_loaded = False
        
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
        
        logger.info(f"VibeVoice STT initialized: model={model_size}, device={self.device}")
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def load_model(self) -> bool:
        """
        Load the Whisper model from HuggingFace.
        
        Returns:
            True if model loaded successfully
        """
        if self._is_loaded:
            return True
        
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            model_id = f"openai/whisper-large-v3-{self.model_size}" if self.model_size != "turbo" else "openai/whisper-large-v3-turbo"
            
            logger.info(f"Loading Whisper model: {model_id}...")
            
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
            
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=self.cache_dir,
            ).to(self.device)
            
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
            )
            
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=self.device,
            )
            
            self._is_loaded = True
            logger.info("Whisper model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, bytes],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio file path, numpy array, or bytes
            sample_rate: Sample rate if audio is numpy array
            language: Optional language code (e.g., 'en', 'es')
            
        Returns:
            Dictionary with:
                - text: Transcribed text
                - language: Detected language
                - duration_ms: Audio duration
        """
        if not self.load_model():
            return {"text": "", "error": "Model not loaded"}
        
        try:
            # Handle different input types
            if isinstance(audio, bytes):
                # Save bytes to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio)
                    audio_path = f.name
                result = self._pipe(audio_path, generate_kwargs={"language": language} if language else {})
                os.unlink(audio_path)
            elif isinstance(audio, np.ndarray):
                # Convert numpy array
                result = self._pipe(
                    {"raw": audio, "sampling_rate": sample_rate},
                    generate_kwargs={"language": language} if language else {},
                )
            else:
                # Assume file path
                result = self._pipe(audio, generate_kwargs={"language": language} if language else {})
            
            return {
                "text": result.get("text", "").strip(),
                "language": language or "auto",
                "model": self.MODEL_ID,
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "error": str(e)}
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        chunk_duration_ms: int = 3000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream transcription for realtime audio.
        
        Args:
            audio_stream: Async iterator of audio chunks (bytes)
            sample_rate: Audio sample rate
            chunk_duration_ms: Process audio in chunks of this duration
            
        Yields:
            Partial transcription results
        """
        if not self.load_model():
            yield {"text": "", "error": "Model not loaded", "is_final": True}
            return
        
        buffer = b""
        bytes_per_chunk = int(sample_rate * 2 * chunk_duration_ms / 1000)  # 16-bit audio
        
        try:
            async for audio_chunk in audio_stream:
                buffer += audio_chunk
                
                if len(buffer) >= bytes_per_chunk:
                    # Convert bytes to numpy
                    audio_array = np.frombuffer(buffer[:bytes_per_chunk], dtype=np.int16).astype(np.float32) / 32768.0
                    buffer = buffer[bytes_per_chunk:]
                    
                    result = self._pipe(
                        {"raw": audio_array, "sampling_rate": sample_rate},
                    )
                    
                    yield {
                        "text": result.get("text", "").strip(),
                        "is_final": False,
                    }
            
            # Process remaining buffer
            if buffer:
                audio_array = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                result = self._pipe(
                    {"raw": audio_array, "sampling_rate": sample_rate},
                )
                yield {
                    "text": result.get("text", "").strip(),
                    "is_final": True,
                }
                
        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            yield {"text": "", "error": str(e), "is_final": True}


class VibeVoiceRealtime:
    """
    Combined STT + TTS for realtime voice conversations.
    
    Provides a unified interface for voice-based interaction with LLMs.
    """
    
    def __init__(
        self,
        speaker: str = "Carter",
        stt_model_size: str = "turbo",
        device: Optional[str] = None,
    ):
        """
        Initialize realtime voice client.
        
        Args:
            speaker: TTS speaker voice
            stt_model_size: Whisper model size for STT
            device: Device to run on
        """
        self.tts = VibeVoiceRealtimeClient(speaker=speaker, device=device)
        self.stt = VibeVoiceSTT(model_size=stt_model_size, device=device)
        self._is_listening = False
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening
    
    def start_listening(self) -> None:
        """Start listening for audio input."""
        self._is_listening = True
        logger.info("Started listening")
    
    def stop_listening(self) -> None:
        """Stop listening for audio input."""
        self._is_listening = False
        logger.info("Stopped listening")
    
    def toggle_listening(self) -> bool:
        """Toggle listening state. Returns new state."""
        self._is_listening = not self._is_listening
        logger.info(f"Listening: {self._is_listening}")
        return self._is_listening
    
    def transcribe(self, audio: Union[str, np.ndarray, bytes], **kwargs) -> Dict[str, Any]:
        """Transcribe audio to text."""
        return self.stt.transcribe(audio, **kwargs)
    
    def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Synthesize text to audio."""
        return self.tts.synthesize(text, **kwargs)
    
    async def voice_turn(
        self,
        audio_input: Union[str, np.ndarray, bytes],
        llm_callback,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Complete voice conversation turn: STT -> LLM -> TTS.
        
        Args:
            audio_input: User's voice input
            llm_callback: Async function that takes text and returns response text
            **kwargs: Additional arguments for STT/TTS
            
        Returns:
            Dictionary with transcription, response, and audio
        """
        # STT
        transcription = self.transcribe(audio_input, **kwargs)
        if transcription.get("error"):
            return {"error": f"STT failed: {transcription['error']}"}
        
        user_text = transcription["text"]
        if not user_text:
            return {"error": "No speech detected"}
        
        # LLM
        try:
            response_text = await llm_callback(user_text)
        except Exception as e:
            return {"error": f"LLM failed: {e}", "user_text": user_text}
        
        # TTS
        audio_result = self.synthesize(response_text, **kwargs)
        if audio_result.get("error"):
            return {
                "user_text": user_text,
                "response_text": response_text,
                "error": f"TTS failed: {audio_result['error']}",
            }
        
        return {
            "user_text": user_text,
            "response_text": response_text,
            "audio": audio_result["audio"],
            "sample_rate": audio_result["sample_rate"],
            "duration_ms": audio_result.get("duration_ms", 0),
        }


async def demo():
    """Demo function for testing VibeVoice."""
    client = VibeVoiceRealtimeClient()
    
    print("Available speakers:", client.get_available_speakers())
    print(f"Using speaker: {client.speaker}")
    print(f"Device: {client.device}")
    
    # Test synthesis
    result = client.synthesize("Hello! This is a test of VibeVoice realtime text to speech.")
    
    if result.get("audio") is not None:
        print(f"Generated {result['duration_ms']:.0f}ms of audio")
        client.save_audio(result["audio"], "vibevoice_test.wav")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(demo())
