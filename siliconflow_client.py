#!/usr/bin/env python3
"""
Comprehensive SiliconFlow API Client
Supports all endpoints: chat, embeddings, rerank, images, audio, video, models
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capabilities and metadata"""

    id: str
    name: str
    context_window: int
    supports_chat: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_embeddings: bool = False
    supports_rerank: bool = False
    supports_function_calling: bool = False
    supports_reasoning: bool = False
    max_tokens: Optional[int] = None
    input_pricing: Optional[float] = None  # per million tokens
    output_pricing: Optional[float] = None  # per million tokens
    input_cached_pricing: Optional[float] = None
    output_cached_pricing: Optional[float] = None


@dataclass
class ChatMessage:
    """Chat message structure"""

    role: str  # "user", "assistant", "system"
    content: Union[str, List[Dict]]  # string or list of content blocks


@dataclass
class ChatRequest:
    """Chat completion request"""

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    min_p: Optional[float] = None
    response_format: Optional[Dict] = None
    tools: Optional[List[Dict]] = None


@dataclass
class EmbeddingRequest:
    """Embedding request"""

    model: str
    input: Union[str, List[str]]
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


@dataclass
class RerankRequest:
    """Rerank request"""

    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = True


@dataclass
class ImageGenerationRequest:
    """Image generation request"""

    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    image_size: str = "1024x1024"
    batch_size: int = 1
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@dataclass
class AudioGenerationRequest:
    """Audio generation request"""

    model: str
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


@dataclass
class VideoGenerationRequest:
    """Video generation request"""

    model: str
    prompt: str
    duration: int = 5
    aspect_ratio: str = "16:9"


class SiliconFlowAPIClient:
    """Comprehensive SiliconFlow API client"""

    BASE_URL = "https://api.siliconflow.com/v1"

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

        # Setup caching
        self.cache_dir = cache_dir or Path.home() / ".cache" / "siliconflow"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(hours=1)  # Cache for 1 hour

    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for request"""
        key_data = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if valid"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time < self.cache_expiry:
                return cached["data"]
            else:
                cache_file.unlink()  # Remove expired cache
        except (json.JSONDecodeError, KeyError, ValueError):
            cache_file.unlink()  # Remove corrupted cache

        return None

    def _cache_response(self, cache_key: str, data: Dict):
        """Cache response"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with caching and error handling"""
        url = f"{self.BASE_URL}{endpoint}"
        cache_key = None

        # Check cache for GET requests
        if method.upper() == "GET":
            cache_key = self._get_cache_key(endpoint, kwargs.get("params") or {})
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.info(f"Using cached response for {endpoint}")
                return cached

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()

            data = response.json()

            # Cache successful GET responses
            if method.upper() == "GET" and cache_key:
                self._cache_response(cache_key, data)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_models(self) -> List[ModelCapabilities]:
        """Get all available models with their capabilities"""
        logger.info("Fetching models from SiliconFlow API...")
        response = self._make_request("GET", "/models")

        models = []
        for model_data in response.get("data", []):
            model_id = model_data.get("id", "")
            model_type = model_data.get("type", "")

            # Determine capabilities based on model ID and type
            capabilities = self._infer_capabilities(model_id, model_type)

            model = ModelCapabilities(
                id=model_id,
                name=self._format_model_name(model_id),
                context_window=self._infer_context_window(model_id),
                supports_chat=capabilities.get("supports_chat", False),
                supports_vision=capabilities.get("supports_vision", False),
                supports_audio=capabilities.get("supports_audio", False),
                supports_video=capabilities.get("supports_video", False),
                supports_embeddings=capabilities.get("supports_embeddings", False),
                supports_rerank=capabilities.get("supports_rerank", False),
                supports_function_calling=self._supports_function_calling(model_id),
                supports_reasoning=self._supports_reasoning(model_id),
                max_tokens=self._infer_max_tokens(model_id),
                input_pricing=self._infer_pricing(model_id, "input"),
                output_pricing=self._infer_pricing(model_id, "output"),
                input_cached_pricing=self._infer_pricing(model_id, "input_cached"),
                output_cached_pricing=self._infer_pricing(model_id, "output_cached"),
            )
            models.append(model)

        logger.info(f"Found {len(models)} models")
        return models

    def chat_completion(self, request: ChatRequest) -> Dict:
        """Create chat completion"""
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "stream": request.stream,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.top_k:
            payload["top_k"] = request.top_k
        if request.stop:
            payload["stop"] = request.stop
        if request.enable_thinking:
            payload["enable_thinking"] = request.enable_thinking
        if request.thinking_budget:
            payload["thinking_budget"] = request.thinking_budget
        if request.min_p:
            payload["min_p"] = request.min_p
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools

        return self._make_request("POST", "/chat/completions", json=payload)

    def create_embeddings(self, request: EmbeddingRequest) -> Dict:
        """Create embeddings"""
        payload = {
            "model": request.model,
            "input": request.input,
            "encoding_format": request.encoding_format,
        }

        if request.dimensions:
            payload["dimensions"] = request.dimensions
        if request.user:
            payload["user"] = request.user

        return self._make_request("POST", "/embeddings", json=payload)

    def create_rerank(self, request: RerankRequest) -> Dict:
        """Create rerank"""
        payload = {
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
            "return_documents": request.return_documents,
        }

        if request.top_n:
            payload["top_n"] = request.top_n

        return self._make_request("POST", "/rerank", json=payload)

    def create_image(self, request: ImageGenerationRequest) -> Dict:
        """Create image"""
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "image_size": request.image_size,
            "batch_size": request.batch_size,
        }

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.seed:
            payload["seed"] = request.seed
        if request.num_inference_steps:
            payload["num_inference_steps"] = request.num_inference_steps
        if request.guidance_scale:
            payload["guidance_scale"] = request.guidance_scale

        return self._make_request("POST", "/images/generations", json=payload)

    def create_speech(self, request: AudioGenerationRequest) -> Dict:
        """Create speech"""
        payload = {
            "model": request.model,
            "input": request.input,
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }

        return self._make_request("POST", "/audio/speech", json=payload)

    def upload_voice(self, voice_file: Path, voice_name: str) -> Dict:
        """Upload reference voice"""
        with open(voice_file, "rb") as f:
            files = {"file": (voice_file.name, f, "audio/wav")}
            data = {"name": voice_name}
            return self._make_request("POST", "/audio/voices", files=files, data=data)

    def list_voices(self) -> Dict:
        """List reference voices"""
        return self._make_request("GET", "/audio/voices")

    def delete_voice(self, voice_id: str) -> Dict:
        """Delete reference voice"""
        return self._make_request("DELETE", f"/audio/voices/{voice_id}")

    def create_transcription(
        self, audio_file: Path, model: str = "FunAudioLLM/SenseVoiceSmall"
    ) -> Dict:
        """Create audio transcription"""
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/wav")}
            data = {"model": model}
            return self._make_request(
                "POST", "/audio/transcriptions", files=files, data=data
            )

    def create_video(self, request: VideoGenerationRequest) -> Dict:
        """Create video"""
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "duration": request.duration,
            "aspect_ratio": request.aspect_ratio,
        }

        return self._make_request("POST", "/videos/submit", json=payload)

    def get_video_status(self, video_id: str) -> Dict:
        """Get video generation status"""
        return self._make_request("GET", f"/videos/{video_id}")

    def get_user_info(self) -> Dict:
        """Get user information"""
        return self._make_request("GET", "/user/info")

    # Helper methods for capability inference
    def _infer_capabilities(self, model_id: str, model_type: str) -> Dict:
        """Infer model capabilities from ID and type"""
        capabilities = {}

        model_lower = model_id.lower()
        type_lower = model_type.lower()

        # Initialize all capabilities to False
        capabilities["supports_chat"] = False
        capabilities["supports_vision"] = False
        capabilities["supports_audio"] = False
        capabilities["supports_video"] = False
        capabilities["supports_embeddings"] = False
        capabilities["supports_rerank"] = False

        # Embedding capabilities - check type first
        capabilities["supports_embeddings"] = "embedding" in type_lower

        # Rerank capabilities - check type first
        capabilities["supports_rerank"] = "rerank" in type_lower

        # Audio capabilities - check both ID and type
        audio_keywords = ["tts", "audio", "speech", "voice"]
        capabilities["supports_audio"] = any(
            keyword in model_lower for keyword in audio_keywords
        ) or any(keyword in type_lower for keyword in audio_keywords)

        # Video capabilities - check both ID and type
        video_keywords = ["t2v", "video", "i2v", "flux"]
        capabilities["supports_video"] = any(
            keyword in model_lower for keyword in video_keywords
        ) or any(keyword in type_lower for keyword in video_keywords)

        # Vision capabilities - check both ID and type
        vision_keywords = ["vl", "vision", "visual", "multimodal"]
        capabilities["supports_vision"] = any(
            keyword in model_lower for keyword in vision_keywords
        ) or any(keyword in type_lower for keyword in vision_keywords)

        # Chat capabilities - more conservative approach
        # Models that are explicitly for other purposes should not be marked as chat
        specialized_keywords = [
            "embedding", "rerank", "tts", "speech", "audio", "video",
            "t2v", "i2v", "flux", "vl", "vision", "visual", "multimodal"
        ]
        
        # Check if model is specialized (has keywords for non-chat tasks)
        is_specialized = any(keyword in model_lower for keyword in specialized_keywords)
        
        # Chat indicators in type
        chat_type_indicators = ["chat", "text", "completion", "instruction", "instruct"]
        has_chat_type = any(keyword in type_lower for keyword in chat_type_indicators)
        
        # Chat indicators in model ID (common chat model patterns)
        chat_id_indicators = [
            "instruct", "chat", "qwen", "deepseek", "glm", "kimi",
            "llama", "mistral", "mixtral", "gemma", "phi", "yi"
        ]
        has_chat_id = any(keyword in model_lower for keyword in chat_id_indicators)
        
        # Determine chat capability
        # Rule 1: If type explicitly indicates chat, mark as chat (unless specialized)
        # Rule 2: If model ID indicates chat AND not specialized, mark as chat
        # Rule 3: If type is unknown/empty and model not specialized, be conservative
        if has_chat_type:
            # Type indicates chat, but check if it's a specialized model
            if not is_specialized:
                capabilities["supports_chat"] = True
            else:
                # Specialized model with chat type (e.g., vision-language model)
                # These often support chat too, so mark as chat
                capabilities["supports_chat"] = True
        elif has_chat_id and not is_specialized:
            # Model ID suggests chat capability and not specialized
            capabilities["supports_chat"] = True
        elif type_lower == "" or type_lower == "unknown":
            # No type information, be conservative
            capabilities["supports_chat"] = False
        else:
            # Has type but not chat type, likely not a chat model
            capabilities["supports_chat"] = False

        return capabilities

    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into human-readable name"""
        parts = model_id.split("/")
        if len(parts) > 1:
            name = parts[1].replace("-", " ").replace("_", " ")
            return f"{parts[0]} {name}"
        return model_id.replace("-", " ").replace("_", " ")

    def _infer_context_window(self, model_id: str) -> int:
        """Infer context window size from model ID"""
        model_lower = model_id.lower()

        # DeepSeek models
        if "deepseek" in model_lower:
            if "r1" in model_lower:
                return 131072
            return 131072  # V3 series

        # Qwen models
        if "qwen" in model_lower:
            if "qwen3" in model_lower:
                return 32768
            elif "qwen2.5" in model_lower:
                if "72b" in model_lower:
                    return 131072
                return 32768
            elif "qwen2-vl" in model_lower:
                return 32768

        # GLM models
        if "glm" in model_lower:
            return 131072 if "4.6" in model_lower else 32768

        # Kimi models
        if "kimi" in model_lower:
            return 131072

        # Default
        return 4096

    def _infer_max_tokens(self, model_id: str) -> Optional[int]:
        """Infer maximum tokens for model"""
        model_lower = model_id.lower()

        if "deepseek-r1" in model_lower:
            return 8192
        elif "qwen3" in model_lower:
            return 8192
        elif "qwen2.5-72b" in model_lower:
            return 8192

        return None

    def _supports_function_calling(self, model_id: str) -> bool:
        """Check if model supports function calling"""
        model_lower = model_id.lower()

        # Most chat models support function calling
        if any(model in model_lower for model in ["qwen", "deepseek", "glm", "kimi"]):
            return True

        return False

    def _supports_reasoning(self, model_id: str) -> bool:
        """Check if model supports reasoning/thinking"""
        model_lower = model_id.lower()

        return any(
            model in model_lower
            for model in ["deepseek-r1", "qwen3", "qwq", "kimi-k2-thinking"]
        )

    def _infer_pricing(self, model_id: str, pricing_type: str) -> Optional[float]:
        """Infer pricing from model ID"""
        # This would need to be updated with actual pricing
        # For now, return None to indicate unknown pricing
        return None


class SiliconFlowAPIError(Exception):
    """SiliconFlow API error"""

    pass


class SiliconFlowRateLimitError(SiliconFlowAPIError):
    """Rate limit exceeded"""

    pass


class SiliconFlowAuthenticationError(SiliconFlowAPIError):
    """Authentication failed"""

    pass
