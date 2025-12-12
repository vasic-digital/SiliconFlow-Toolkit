#!/usr/bin/env python3
"""
Comprehensive Test Suite for SiliconFlow Toolkit
Tests all API endpoints, model discovery, and configuration building
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from siliconflow_client import (
    SiliconFlowAPIClient,
    SiliconFlowAPIError,
    ModelCapabilities,
    ChatRequest,
    ChatMessage,
    EmbeddingRequest,
    RerankRequest,
    ImageGenerationRequest,
    AudioGenerationRequest,
    VideoGenerationRequest,
)
from model_discovery import SiliconFlowModelDiscovery, ModelRegistry, ModelCategory
from install import EnhancedSiliconFlowConfigBuilder


class TestSiliconFlowAPIClient(unittest.TestCase):
    """Test SiliconFlow API client functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.client = SiliconFlowAPIClient(self.api_key)

    def test_initialization(self):
        """Test client initialization"""
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.BASE_URL, "https://api.siliconflow.com/v1")
        self.assertIn("Authorization", self.client.session.headers)

    @patch("siliconflow_client.requests.Session.request")
    def test_get_models_success(self, mock_request):
        """Test successful model fetching"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "Qwen/Qwen2.5-7B-Instruct", "type": "text-generation"},
                {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "type": "vision"},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        models = self.client.get_models()

        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIsInstance(models[0], ModelCapabilities)

    @patch("siliconflow_client.requests.Session.request")
    def test_chat_completion(self, mock_request):
        """Test chat completion"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"total_tokens": 10},
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        request = ChatRequest(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        response = self.client.chat_completion(request)

        self.assertIn("id", response)
        self.assertIn("choices", response)
        mock_request.assert_called_once()

    @patch("siliconflow_client.requests.Session.request")
    def test_create_embeddings(self, mock_request):
        """Test embedding creation"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        request = EmbeddingRequest(
            model="text-embedding-ada-002", input=["Hello world"]
        )

        response = self.client.create_embeddings(request)

        self.assertIn("data", response)
        mock_request.assert_called_once()

    @patch("siliconflow_client.requests.Session.request")
    def test_create_rerank(self, mock_request):
        """Test reranking"""
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"index": 0, "score": 0.95}]}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        request = RerankRequest(
            model="rerank-model", query="test query", documents=["doc1", "doc2"]
        )

        response = self.client.create_rerank(request)

        self.assertIn("results", response)
        mock_request.assert_called_once()

    @patch("siliconflow_client.requests.Session.request")
    def test_create_image(self, mock_request):
        """Test image generation"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "images": [{"url": "https://example.com/image.png"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        request = ImageGenerationRequest(
            model="image-generator", prompt="A beautiful sunset"
        )

        response = self.client.create_image(request)

        self.assertIn("images", response)
        mock_request.assert_called_once()

    @patch("siliconflow_client.requests.Session.request")
    def test_create_speech(self, mock_request):
        """Test speech synthesis"""
        mock_response = Mock()
        mock_response.content = b"audio data"
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        request = AudioGenerationRequest(
            model="tts-model", input="Hello world", voice="alloy"
        )

        response = self.client.create_speech(request)

        mock_request.assert_called_once()

    def test_caching(self):
        """Test response caching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            client = SiliconFlowAPIClient(self.api_key, cache_dir)

            # Test cache key generation
            key1 = client._get_cache_key("test_endpoint", {"param": "value"})
            key2 = client._get_cache_key("test_endpoint", {"param": "value"})
            self.assertEqual(key1, key2)

            # Test cache storage and retrieval
            test_data = {"test": "data"}
            client._cache_response(key1, test_data)

            cached = client._get_cached_response(key1)
            self.assertEqual(cached, test_data)


class TestSiliconFlowModelDiscovery(unittest.TestCase):
    """Test model discovery functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.discovery = SiliconFlowModelDiscovery(self.api_key)

    @patch("model_discovery.SiliconFlowAPIClient.get_models")
    def test_discover_all_models(self, mock_get_models):
        """Test model discovery"""
        mock_models = [
            ModelCapabilities(
                id="Qwen/Qwen2.5-7B-Instruct",
                name="Qwen 2.5 7B",
                context_window=32768,
                supports_chat=True,
            ),
            ModelCapabilities(
                id="Qwen/Qwen2.5-VL-7B-Instruct",
                name="Qwen 2.5 VL 7B",
                context_window=32768,
                supports_vision=True,
            ),
        ]
        mock_get_models.return_value = mock_models

        registry = self.discovery.discover_all_models()

        self.assertIsInstance(registry, ModelRegistry)
        self.assertGreater(len(registry.chat_models.models), 0)
        self.assertGreater(len(registry.vision_models.models), 0)

    def test_categorize_models(self):
        """Test model categorization"""
        models = [
            ModelCapabilities(
                id="chat-model",
                name="Chat Model",
                context_window=4096,
                supports_chat=True,
            ),
            ModelCapabilities(
                id="vision-model",
                name="Vision Model",
                context_window=4096,
                supports_vision=True,
            ),
            ModelCapabilities(
                id="audio-model",
                name="Audio Model",
                context_window=4096,
                supports_audio=True,
            ),
        ]

        categories = self.discovery._categorize_models(models)

        self.assertIn("chat", categories)
        self.assertIn("vision", categories)
        self.assertIn("audio", categories)
        self.assertEqual(len(categories["chat"].models), 1)
        self.assertEqual(len(categories["vision"].models), 1)
        self.assertEqual(len(categories["audio"].models), 1)

    def test_validate_model_compatibility(self):
        """Test model compatibility validation"""
        with patch.object(self.discovery, "get_model_info") as mock_get_info:
            mock_get_info.return_value = ModelCapabilities(
                id="test-model",
                name="Test Model",
                context_window=4096,
                supports_chat=True,
                supports_function_calling=True,
            )

            # Test valid capabilities
            self.assertTrue(
                self.discovery.validate_model_compatibility(
                    "test-model", ["chat", "function_calling"]
                )
            )

            # Test invalid capabilities
            self.assertFalse(
                self.discovery.validate_model_compatibility("test-model", ["vision"])
            )

    def test_get_models_by_capability(self):
        """Test filtering models by capability"""
        with patch.object(self.discovery, "discover_all_models") as mock_discover:
            registry = ModelRegistry(
                chat_models=ModelCategory(
                    "Chat",
                    "Chat models",
                    [
                        ModelCapabilities(
                            id="chat1",
                            name="Chat 1",
                            context_window=4096,
                            supports_chat=True,
                        ),
                        ModelCapabilities(
                            id="chat2",
                            name="Chat 2",
                            context_window=4096,
                            supports_chat=True,
                        ),
                    ],
                ),
                vision_models=ModelCategory(
                    "Vision",
                    "Vision models",
                    [
                        ModelCapabilities(
                            id="vision1",
                            name="Vision 1",
                            context_window=4096,
                            supports_vision=True,
                        )
                    ],
                ),
                audio_models=ModelCategory("Audio", "Audio models", []),
                video_models=ModelCategory("Video", "Video models", []),
                embedding_models=ModelCategory("Embedding", "Embedding models", []),
                rerank_models=ModelCategory("Rerank", "Rerank models", []),
                last_updated="2024-01-01T00:00:00",
            )
            mock_discover.return_value = registry

            chat_models = self.discovery.get_models_by_capability("chat")
            self.assertEqual(len(chat_models), 2)

            vision_models = self.discovery.get_models_by_capability("vision")
            self.assertEqual(len(vision_models), 1)


class TestEnhancedSiliconFlowConfigBuilder(unittest.TestCase):
    """Test enhanced configuration builder"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.builder = EnhancedSiliconFlowConfigBuilder(self.api_key)

    @patch("install.SiliconFlowModelDiscovery.discover_all_models")
    def test_build_opencode_config(self, mock_discover):
        """Test OpenCode config building"""
        registry = ModelRegistry(
            chat_models=ModelCategory(
                "Chat",
                "Chat models",
                [
                    ModelCapabilities(
                        id="Qwen/Qwen2.5-7B-Instruct",
                        name="Qwen 2.5 7B",
                        context_window=32768,
                        supports_chat=True,
                        supports_function_calling=True,
                    )
                ],
            ),
            vision_models=ModelCategory("Vision", "Vision models", []),
            audio_models=ModelCategory("Audio", "Audio models", []),
            video_models=ModelCategory("Video", "Video models", []),
            embedding_models=ModelCategory("Embedding", "Embedding models", []),
            rerank_models=ModelCategory("Rerank", "Rerank models", []),
            last_updated="2024-01-01T00:00:00",
        )
        mock_discover.return_value = registry

        config = self.builder.build_opencode_config()

        self.assertIn("$schema", config)
        self.assertIn("provider", config)
        self.assertIn("siliconflow", config["provider"])
        self.assertIn("models", config["provider"]["siliconflow"])

    @patch("install.SiliconFlowModelDiscovery.discover_all_models")
    def test_build_crush_config(self, mock_discover):
        """Test Crush config building"""
        registry = ModelRegistry(
            chat_models=ModelCategory(
                "Chat",
                "Chat models",
                [
                    ModelCapabilities(
                        id="Qwen/Qwen2.5-7B-Instruct",
                        name="Qwen 2.5 7B",
                        context_window=32768,
                        supports_chat=True,
                    )
                ],
            ),
            vision_models=ModelCategory("Vision", "Vision models", []),
            audio_models=ModelCategory("Audio", "Audio models", []),
            video_models=ModelCategory("Video", "Video models", []),
            embedding_models=ModelCategory("Embedding", "Embedding models", []),
            rerank_models=ModelCategory("Rerank", "Rerank models", []),
            last_updated="2024-01-01T00:00:00",
        )
        mock_discover.return_value = registry

        config = self.builder.build_crush_config()

        self.assertIn("$schema", config)
        self.assertIn("providers", config)
        self.assertIn("metadata", config)

    def test_merge_opencode_configs(self):
        """Test OpenCode config merging"""
        existing = {
            "theme": "light",
            "model": "old-model",
            "provider": {"other": {"apiKey": "old-key"}},
        }

        new = {
            "$schema": "https://opencode.ai/config.json",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "provider": {
                "siliconflow": {
                    "name": "SiliconFlow",
                    "options": {
                        "apiKey": "new-key",
                        "baseURL": "https://api.siliconflow.com/v1"
                    },
                    "models": {"model1": {}},
                }
            },
        }

        merged = self.builder.merge_configs_smartly(existing, new, "opencode")

        # Should preserve user theme
        self.assertEqual(merged["theme"], "light")
        # Should update to new model
        self.assertEqual(merged["model"], "Qwen/Qwen2.5-7B-Instruct")
        # Should have both providers
        self.assertIn("siliconflow", merged["provider"])
        self.assertIn("other", merged["provider"])

    def test_merge_crush_configs(self):
        """Test Crush config merging"""
        existing = {
            "defaultProvider": "old-provider",
            "defaultModel": "old-model",
            "providers": {"old-provider": {"name": "Old Provider"}},
        }

        new = {
            "$schema": "https://charm.land/crush.json",
            "providers": {
                "siliconflow-chat": {
                    "name": "SiliconFlow Chat",
                    "models": [{"id": "new-model"}],
                }
            },
            "defaultProvider": "siliconflow-chat",
            "defaultModel": "new-model",
        }

        merged = self.builder.merge_configs_smartly(existing, new, "crush")

        # Should have both providers
        self.assertIn("old-provider", merged["providers"])
        self.assertIn("siliconflow-chat", merged["providers"])
        # Should update defaults
        self.assertEqual(merged["defaultProvider"], "siliconflow-chat")
        self.assertEqual(merged["defaultModel"], "new-model")


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            self.skipTest("SILICONFLOW_API_KEY environment variable not set")

    def test_real_api_connection(self):
        """Test real API connection (requires API key)"""
        client = SiliconFlowAPIClient(self.api_key)

        # Test model fetching
        models = client.get_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

        # Test model discovery
        discovery = SiliconFlowModelDiscovery(self.api_key)
        registry = discovery.discover_all_models()
        self.assertIsInstance(registry, ModelRegistry)

        # Test config building
        builder = EnhancedSiliconFlowConfigBuilder(self.api_key)
        opencode_config = builder.build_opencode_config()
        crush_config = builder.build_crush_config()

        self.assertIn("provider", opencode_config)
        self.assertIn("providers", crush_config)

    def test_config_validation(self):
        """Test configuration validation"""
        builder = EnhancedSiliconFlowConfigBuilder("test-key")

        # Test with mock registry
        with patch.object(builder.discovery, "discover_all_models") as mock_discover:
            registry = ModelRegistry(
                chat_models=ModelCategory(
                    "Chat",
                    "Chat models",
                    [
                        ModelCapabilities(
                            id="chat1",
                            name="Chat 1",
                            context_window=4096,
                            supports_chat=True,
                        ),
                        ModelCapabilities(
                            id="chat2",
                            name="Chat 2",
                            context_window=4096,
                            supports_chat=True,
                        ),
                    ],
                ),
                vision_models=ModelCategory(
                    "Vision",
                    "Vision models",
                    [
                        ModelCapabilities(
                            id="vision1",
                            name="Vision 1",
                            context_window=4096,
                            supports_vision=True,
                        )
                    ],
                ),
                audio_models=ModelCategory("Audio", "Audio models", []),
                video_models=ModelCategory("Video", "Video models", []),
                embedding_models=ModelCategory("Embedding", "Embedding models", []),
                rerank_models=ModelCategory("Rerank", "Rerank models", []),
                last_updated="2024-01-01T00:00:00",
            )
            mock_discover.return_value = registry

            config = builder.build_opencode_config()

            # Validate required fields
            self.assertIn("$schema", config)
            self.assertIn("provider", config)
            self.assertIn("siliconflow", config["provider"])


if __name__ == "__main__":
    # Add verbose output
    unittest.main(verbosity=2)
