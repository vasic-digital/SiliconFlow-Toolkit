#!/usr/bin/env python3
"""
SiliconFlow Ultimate Configuration Builder
Dynamically fetches ALL models and builds complete configurations
"""
import json
import requests
import os
import sys
from pathlib import Path
from datetime import datetime
import getpass

class SiliconFlowConfigBuilder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.com/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        # Model type mappings from documentation
        self.model_categories = {
            "chat": {
                "endpoint": "/chat/completions",
                "capabilities": ["text-generation", "tool-calling", "reasoning"],
                "params": self._get_chat_params()
            },
            "embedding": {
                "endpoint": "/embeddings",
                "capabilities": ["text-embedding", "semantic-search"],
                "params": self._get_embedding_params()
            },
            "reranker": {
                "endpoint": "/rerank",
                "capabilities": ["document-reranking", "relevance-scoring"],
                "params": self._get_rerank_params()
            },
            "text-to-image": {
                "endpoint": "/images/generations",  # Inferred from API reference
                "capabilities": ["image-generation"],
                "params": self._get_image_params()
            },
            "image-to-image": {
                "endpoint": "/images/edits",  # Inferred
                "capabilities": ["image-editing", "image-transformation"],
                "params": {}
            }
        }
    
    def _get_chat_params(self):
        """Extract all chat completion parameters from docs"""
        return {
            "stream": {"type": "boolean", "default": False},
            "max_tokens": {"type": "integer", "default": 4096},
            "enable_thinking": {"type": "boolean", "default": False},
            "thinking_budget": {"type": "integer", "range": [128, 32768], "default": 4096},
            "min_p": {"type": "float", "range": [0, 1], "default": 0.05},
            "stop": {"type": "array", "default": []},
            "temperature": {"type": "float", "default": 0.7},
            "top_p": {"type": "float", "default": 0.7},
            "top_k": {"type": "integer", "default": 50},
            "frequency_penalty": {"type": "float", "default": 0.5},
            "n": {"type": "integer", "default": 1},
            "response_format": {"type": "object", "default": {"type": "text"}},
            "tools": {"type": "array", "max_items": 128},
            "tool_choice": {"type": "object", "default": {"type": "auto", "disable_parallel_tool_use": True}}
        }
    
    def _get_embedding_params(self):
        """Extract embedding parameters"""
        return {
            "encoding_format": {"type": "string", "options": ["float", "base64"], "default": "float"},
            "dimensions": {"type": "integer", "options": [64, 128, 256, 512, 768, 1024, 2048, 4096]}
        }
    
    def _get_rerank_params(self):
        """Extract rerank parameters"""
        return {
            "top_n": {"type": "integer", "default": 4},
            "return_documents": {"type": "boolean", "default": True},
            "max_chunks_per_doc": {"type": "integer"},
            "overlap_tokens": {"type": "integer", "max": 80}
        }
    
    def _get_image_params(self):
        """Image generation parameters (inferred structure)"""
        return {
            "size": {"type": "string", "options": ["256x256", "512x512", "1024x1024"]},
            "quality": {"type": "string", "options": ["standard", "hd"]},
            "style": {"type": "string", "options": ["vivid", "natural"]}
        }
    
    def fetch_all_models(self):
        """Fetch ALL models from SiliconFlow API with all types"""
        all_models = []
        
        # Model types from documentation
        model_types = ["text", "image", "audio", "video"]
        model_subtypes = ["chat", "embedding", "reranker", "text-to-image", 
                         "image-to-image", "speech-to-text", "text-to-video"]
        
        print("🔄 Fetching models from SiliconFlow API...")
        
        # Try to fetch all models (no filter)
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                all_models = data.get("data", [])
                print(f"✅ Found {len(all_models)} total models")
                
                # If API returns empty or limited, use known models from docs
                if len(all_models) < 10:
                    print("⚠️  API returned limited models, using documented models")
                    all_models = self._get_documented_models()
            else:
                print(f"⚠️  API returned {response.status_code}, using documented models")
                all_models = self._get_documented_models()
                
        except Exception as e:
            print(f"⚠️  Error fetching models: {e}, using documented models")
            all_models = self._get_documented_models()
        
        # Categorize models by inferred type
        categorized = self._categorize_models(all_models)
        return categorized
    
    def _get_documented_models(self):
        """Fallback to models explicitly documented"""
        documented_models = []
        
        # Chat models from chat completions docs
        chat_models = [
            "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3.2",
            "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct",
            "moonshotai/Kimi-K2-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct",
            "baidu/ERNIE-4.5-300B-A47B", "zai-org/GLM-4.6",
            "tencent/Hunyuan-A13B-Instruct", "Qwen/Qwen3-235B-A22B-Instruct-2507"
        ]
        
        # Embedding models
        embedding_models = [
            "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-large-zh-v1.5"
        ]
        
        # Reranker models
        reranker_models = [
            "Qwen/Qwen3-Reranker-8B", "Qwen/Qwen3-Reranker-4B",
            "Qwen/Qwen3-Reranker-0.6B", "BAAI/bge-reranker-v2-m3"
        ]
        
        # Add all with placeholder objects
        for model_id in chat_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "chat"
            })
        
        for model_id in embedding_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "embedding"
            })
        
        for model_id in reranker_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "reranker"
            })
        
        return documented_models
    
    def _categorize_models(self, models):
        """Categorize models by their likely function based on name patterns"""
        categorized = {
            "chat": [], "embedding": [], "reranker": [],
            "vision": [], "audio": [], "image": [], "video": []
        }
        
        for model in models:
            model_id = model.get("id", "").lower()
            
            # Enhanced pattern matching
            if any(keyword in model_id for keyword in ["embedding", "bge-", "embed"]):
                model["category"] = "embedding"
                categorized["embedding"].append(model)
            elif any(keyword in model_id for keyword in ["rerank", "reranker"]):
                model["category"] = "reranker"
                categorized["reranker"].append(model)
            elif any(keyword in model_id for keyword in ["vl-", "vision", "visual", "-vl", "glm-4.5v", "glm-4.6v"]):
                model["category"] = "vision"
                categorized["vision"].append(model)
            elif any(keyword in model_id for keyword in ["stable-diffusion", "dalle", "midjourney", "imagen"]):
                model["category"] = "image"
                categorized["image"].append(model)
            elif any(keyword in model_id for keyword in ["whisper", "speech", "audio", "tts"]):
                model["category"] = "audio"
                categorized["audio"].append(model)
            elif any(keyword in model_id for keyword in ["video", "sora", "gen-2"]):
                model["category"] = "video"
                categorized["video"].append(model)
            else:
                # Default to chat for most models
                model["category"] = "chat"
                categorized["chat"].append(model)
        
        return categorized
    
    def build_crush_config(self, categorized_models):
        """Build comprehensive Crush configuration"""
        
        # Build provider configurations for each category
        providers = {}
        
        # Chat Provider (Primary)
        if categorized_models["chat"]:
            providers["siliconflow-chat"] = {
                "name": "SiliconFlow Chat Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/chat/completions",
                "capabilities": ["chat", "reasoning", "tool-calling", "function-calling"],
                "parameters": self._get_chat_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "context_window": self._infer_context_window(model["id"]),
                        "default_max_tokens": 32000,
                        "can_reason": self._supports_reasoning(model["id"]),
                        "supports_attachments": "vl" in model["id"].lower() or "vision" in model["id"].lower(),
                        "supports_tools": True,
                        "thinking_enabled": self._supports_thinking(model["id"]),
                        "max_thinking_budget": 32768,
                        "capabilities": self._infer_capabilities(model["id"])
                    }
                    for model in categorized_models["chat"][:20]  # Limit to top 20 chat models
                ]
            }
        
        # Embedding Provider
        if categorized_models["embedding"]:
            providers["siliconflow-embedding"] = {
                "name": "SiliconFlow Embeddings",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/embeddings",
                "capabilities": ["text-embedding", "semantic-search", "vector-search"],
                "parameters": self._get_embedding_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "max_input_tokens": self._infer_embedding_tokens(model["id"]),
                        "dimensions": self._infer_embedding_dimensions(model["id"]),
                        "encoding_formats": ["float", "base64"],
                        "capabilities": ["embedding", "similarity-search"]
                    }
                    for model in categorized_models["embedding"]
                ]
            }
        
        # Reranker Provider
        if categorized_models["reranker"]:
            providers["siliconflow-reranker"] = {
                "name": "SiliconFlow Rerankers",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/rerank",
                "capabilities": ["document-reranking", "relevance-scoring"],
                "parameters": self._get_rerank_params(),
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "max_documents": 100,
                        "return_documents": True,
                        "chunking_support": "bge-reranker" in model["id"],
                        "capabilities": ["reranking", "relevance-scoring"]
                    }
                    for model in categorized_models["reranker"]
                ]
            }
        
        # Vision Provider
        if categorized_models["vision"]:
            providers["siliconflow-vision"] = {
                "name": "SiliconFlow Vision Language Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/chat/completions",
                "capabilities": ["vision", "image-understanding", "multimodal"],
                "parameters": {**self._get_chat_params(), **self._get_vision_params()},
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "context_window": 131072,
                        "supports_images": True,
                        "max_image_size": "2048x2048",
                        "image_formats": ["png", "jpeg", "jpg", "gif", "webp"],
                        "capabilities": ["vision", "ocr", "image-description", "visual-reasoning"]
                    }
                    for model in categorized_models["vision"]
                ]
            }
        
        # Build complete Crush config
        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": "siliconflow-chat" if "siliconflow-chat" in providers else list(providers.keys())[0],
            "defaultModel": categorized_models["chat"][0]["id"] if categorized_models["chat"] else "",
            
            # Advanced sampling with all documented parameters
            "sampling": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 32000,
                "enable_thinking": False,
                "thinking_budget": 4096,
                "min_p": 0.05,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.0,
                "stop_sequences": [],
                "n": 1,
                "stream": True,
                "response_format": {"type": "text"}
            },
            
            # Comprehensive features
            "features": {
                "lsp_integration": {
                    "enabled": True,
                    "auto_completion": True,
                    "code_actions": True,
                    "diagnostics": True,
                    "hover": True,
                    "signature_help": True,
                    "definition": True,
                    "references": True
                },
                "mcp_tools": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "filesystem",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/media/milosvasic/DATA4TB/Projects"]
                        },
                        {
                            "name": "sqlite",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-sqlite"]
                        },
                        {
                            "name": "github",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-github"]
                        },
                        {
                            "name": "web-search",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-web-search"]
                        },
                        {
                            "name": "brave-search",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-brave-search"]
                        }
                    ]
                },
                "session_memory": True,
                "session_caching": True,
                "token_usage": True,
                "reasoning_traces": True,
                "tool_calling": True,
                "function_calling": {
                    "enabled": True,
                    "max_functions": 128,
                    "strict_schema": False,
                    "parallel_tool_use": True
                },
                "embeddings": True,
                "reranking": True,
                "vision": len(categorized_models["vision"]) > 0,
                "audio": len(categorized_models["audio"]) > 0,
                "image_generation": len(categorized_models["image"]) > 0,
                "video_generation": len(categorized_models["video"]) > 0
            },
            
            # Model routing based on task
            "model_routing": {
                "coding": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "debugging": "deepseek-ai/DeepSeek-V3.2",
                "reasoning": "moonshotai/Kimi-K2-Instruct",
                "vision": "Qwen/Qwen2.5-VL-72B-Instruct" if categorized_models["vision"] else None,
                "chat": "Qwen/Qwen2.5-7B-Instruct",
                "embedding": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                "reranking": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                "default": "Qwen/Qwen2.5-72B-Instruct"
            },
            
            # Advanced limits
            "limits": {
                "max_input_tokens": 190000,
                "max_output_tokens": 32000,
                "max_context_window": 200000,
                "max_images_per_request": 10,
                "max_tools_per_request": 128,
                "max_embedding_tokens": 32768,
                "max_rerank_documents": 100
            },
            
            # UI configuration
            "ui": {
                "theme": "dark",
                "show_model_info": True,
                "show_token_counts": True,
                "show_reasoning_traces": True,
                "show_tool_calls": True,
                "log_level": "debug"
            },
            
            # Agent configurations
            "agents": {
                "coder": {
                    "system_prompt": "You are an expert coding assistant. Use reasoning traces for complex problems. Prioritize clean, testable code with error handling. Use available tools and MCP servers when appropriate.",
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "temperature": 0.1,
                    "enable_thinking": True,
                    "thinking_budget": 8192
                },
                "debugger": {
                    "system_prompt": "Debug systematically. Analyze stack traces, suggest fixes with diffs, write unit tests. Use low temperature and step-by-step reasoning.",
                    "model": "deepseek-ai/DeepSeek-V3.2",
                    "temperature": 0.1,
                    "enable_thinking": True
                },
                "researcher": {
                    "system_prompt": "Research and analyze information. Use web search tools, cite sources, provide comprehensive answers with references.",
                    "model": "moonshotai/Kimi-K2-Instruct",
                    "temperature": 0.3
                },
                "vision_analyst": {
                    "system_prompt": "Analyze images and visual content. Describe scenes, extract text, identify objects, provide detailed visual analysis.",
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct" if categorized_models["vision"] else "Qwen/Qwen2.5-72B-Instruct",
                    "temperature": 0.2
                }
            },
            
            # Tool configurations
            "tools": {
                "enabled": True,
                "auto_tool_choice": True,
                "max_parallel_tools": 5,
                "timeout": 30000,
                "validation": {
                    "validate_arguments": True,
                    "allow_partial": False,
                    "strict_mode": False
                }
            },
            
            # Embedding configurations
            "embeddings": {
                "enabled": True,
                "default_model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                "encoding_format": "float",
                "dimensions": 1024,
                "normalize": True,
                "batch_size": 32
            },
            
            # Reranking configurations
            "reranking": {
                "enabled": True,
                "default_model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                "top_n": 10,
                "return_documents": True,
                "score_threshold": 0.5
            },
            
            # Options
            "options": {
                "debug": True,
                "debug_lsp": False,
                "auto_save_session": True,
                "auto_update_models": True,
                "cache_embeddings": True,
                "prefer_native_tools": True,
                "fallback_to_local": True,
                "timeout": 60000,
                "retry_attempts": 3
            }
        }
        
        return config
    
    def build_opencode_config(self, categorized_models):
        """Build comprehensive OpenCode configuration"""
        
        # Build models object for OpenCode
        models = {}
        for category, model_list in categorized_models.items():
            for model in model_list[:10]:  # Top 10 per category
                model_id = model["id"]
                models[model_id] = {
                    "name": self._format_model_name(model_id),
                    "category": category,
                    "capabilities": self._infer_capabilities(model_id),
                    "contextWindow": self._infer_context_window(model_id),
                    "supportsFunctionCalling": category in ["chat", "vision"],
                    "supportsVision": category in ["vision", "image"],
                    "supportsAudio": category == "audio",
                    "supportsVideo": category == "video",
                    "maxTokens": 32000 if category == "chat" else None,
                    "recommendedFor": self._recommended_for(category, model_id)
                }
        
        config = {
            "$schema": "https://opencode.ai/config.json",
            "theme": "dark",
            "model": categorized_models["chat"][0]["id"] if categorized_models["chat"] else list(models.keys())[0],
            "small_model": "Qwen/Qwen2.5-7B-Instruct",
            "coding_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "reasoning_model": "moonshotai/Kimi-K2-Instruct",
            "vision_model": categorized_models["vision"][0]["id"] if categorized_models["vision"] else None,
            "embedding_model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
            "reranker_model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
            
            # Comprehensive provider configuration
            "provider": {
                "siliconflow": {
                    "name": "SiliconFlow Full Stack",
                    "npm": "@ai-sdk/openai-compatible",
                    "models": models,
                    "options": {
                        "apiKey": self.api_key,
                        "baseURL": self.base_url,
                        "defaultHeaders": {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        },
                        "timeout": 60000,
                        "maxRetries": 3
                    },
                    "endpoints": {
                        "chat": "/chat/completions",
                        "embeddings": "/embeddings",
                        "rerank": "/rerank",
                        "images": "/images/generations",
                        "audio": "/audio/transcriptions"
                    }
                }
            },
            
            # Complete instructions with all capabilities
            "instructions": [
                "You are a full-stack AI assistant powered by SiliconFlow with access to multiple capabilities:",
                "1. Text generation with reasoning and tool-calling support",
                "2. Vision and image understanding (when vision models available)",
                "3. Embeddings for semantic search and vector operations",
                "4. Reranking for document relevance scoring",
                "5. Function calling with up to 128 tools simultaneously",
                "6. MCP integration for filesystem, SQLite, GitHub, and web search",
                "Use appropriate capabilities based on the task. Always reason step-by-step for complex problems.",
                "For coding: prioritize clean, testable, production-ready code.",
                "For analysis: use available tools and cite sources when appropriate.",
                "For vision tasks: describe images thoroughly and extract relevant information."
            ],
            
            # Tool integrations
            "tools": {
                "fileSystem": True,
                "terminal": True,
                "git": True,
                "browser": False,
                "mcp": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "filesystem",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/media/milosvasic/DATA4TB/Projects"],
                            "capabilities": ["read", "write", "search"]
                        },
                        {
                            "name": "sqlite",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                            "capabilities": ["query", "schema", "tables"]
                        },
                        {
                            "name": "github",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-github"],
                            "env": {
                                "GITHUB_TOKEN": "${env:GITHUB_TOKEN}"
                            }
                        },
                        {
                            "name": "web-search",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-web-search"]
                        }
                    ]
                },
                "embeddings": {
                    "enabled": True,
                    "provider": "siliconflow",
                    "model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                    "autoEmbed": True
                },
                "reranking": {
                    "enabled": True,
                    "provider": "siliconflow",
                    "model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None
                }
            },
            
            # API parameters from documentation
            "parameters": {
                "chat": self._get_chat_params(),
                "embedding": self._get_embedding_params(),
                "rerank": self._get_rerank_params()
            },
            
            # Completion configuration
            "completion": {
                "enabled": True,
                "provider": "siliconflow",
                "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "temperature": 0.1,
                "maxTokens": 512,
                "enable_thinking": False,
                "thinking_budget": 2048
            },
            
            # Chat configuration
            "chat": {
                "enabled": True,
                "provider": "siliconflow",
                "defaultModel": "Qwen/Qwen2.5-72B-Instruct",
                "temperature": 0.7,
                "maxTokens": 8192,
                "stream": True,
                "enable_thinking": True,
                "thinking_budget": 4096,
                "tools": True,
                "tool_choice": "auto"
            },
            
            # Session management
            "session": {
                "memoryEnabled": True,
                "maxMessages": 100,
                "autoSummarize": True,
                "embedMessages": True,
                "similarityThreshold": 0.85
            },
            
            # UI configuration
            "ui": {
                "showTokenCounts": True,
                "showReasoningTraces": True,
                "showToolCalls": True,
                "showEmbeddingVectors": False,
                "showRerankScores": True,
                "autoFormatCode": True,
                "theme": "dark",
                "fontSize": 14,
                "lineHeight": 1.6
            },
            
            # Keybindings
            "keybindings": {
                "toggleChat": "ctrl+shift+c",
                "quickCompletion": "ctrl+space",
                "explainCode": "ctrl+shift+e",
                "refactorCode": "ctrl+shift+r",
                "debugCode": "ctrl+shift+d",
                "visionAnalysis": "ctrl+shift+v",
                "embedSelection": "ctrl+shift+b",
                "rerankContext": "ctrl+shift+k"
            },
            
            # Advanced features
            "features": {
                "autoModelSelection": True,
                "contextAwareRouting": True,
                "intentRecognition": True,
                "costOptimization": True,
                "fallbackStrategies": True,
                "performanceMonitoring": True,
                "usageAnalytics": True
            },
            
            # Limits
            "limits": {
                "maxContextTokens": 200000,
                "maxCompletionTokens": 32000,
                "maxEmbeddingTokens": 32768,
                "maxRerankDocuments": 100,
                "maxImagesPerRequest": 10,
                "maxToolsPerRequest": 128,
                "rateLimit": {
                    "requestsPerMinute": 60,
                    "tokensPerMinute": 100000
                }
            },
            
            # Cache configuration
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "maxSize": 1000,
                "embeddingCache": True,
                "completionCache": True
            }
        }
        
        return config
    
    # Helper methods
    def _format_model_name(self, model_id):
        """Format model ID for display"""
        parts = model_id.split("/")
        if len(parts) > 1:
            return f"{parts[0]} {parts[1]}"
        return model_id
    
    def _infer_context_window(self, model_id):
        """Infer context window from model name"""
        model_id_lower = model_id.lower()
        if "128k" in model_id_lower or "200000" in model_id:
            return 200000
        elif "32k" in model_id_lower:
            return 32768
        elif "16k" in model_id_lower:
            return 16384
        elif "8k" in model_id_lower:
            return 8192
        elif "qwen3" in model_id_lower:
            return 131072
        elif "qwen2.5" in model_id_lower:
            return 32768
        else:
            return 4096  # Default
    
    def _supports_reasoning(self, model_id):
        """Check if model supports reasoning"""
        return any(keyword in model_id.lower() for keyword in 
                  ["r1", "thinking", "k2", "reason", "deepseek-r1"])
    
    def _supports_thinking(self, model_id):
        """Check if model supports thinking mode"""
        return any(keyword in model_id.lower() for keyword in 
                  ["qwen3", "hunyuan", "glm-4.6v", "glm-4.5v", "deepseek-v3"])
    
    def _infer_capabilities(self, model_id):
        """Infer model capabilities"""
        caps = []
        model_id_lower = model_id.lower()
        
        if "coder" in model_id_lower or "code" in model_id_lower:
            caps.append("coding")
        if "vl" in model_id_lower or "vision" in model_id_lower:
            caps.append("vision")
        if "embedding" in model_id_lower:
            caps.append("embedding")
        if "reranker" in model_id_lower:
            caps.append("reranking")
        if any(keyword in model_id_lower for keyword in ["r1", "thinking", "reason"]):
            caps.append("reasoning")
        if "chat" in model_id_lower or "instruct" in model_id_lower:
            caps.append("chat")
        
        return caps if caps else ["general"]
    
    def _infer_embedding_tokens(self, model_id):
        """Infer max embedding tokens"""
        if "qwen3-embedding-8b" in model_id.lower():
            return 32768
        elif "bge-large" in model_id.lower():
            return 512
        elif "bge-m3" in model_id.lower():
            return 8192
        else:
            return 2048
    
    def _infer_embedding_dimensions(self, model_id):
        """Infer embedding dimensions"""
        if "8b" in model_id.lower():
            return 1024
        elif "4b" in model_id.lower():
            return 768
        elif "0.6b" in model_id.lower():
            return 512
        else:
            return 1024
    
    def _recommended_for(self, category, model_id):
        """Get recommended use cases"""
        recommendations = {
            "chat": ["general-conversation", "analysis", "writing"],
            "embedding": ["semantic-search", "clustering", "similarity"],
            "reranker": ["document-ranking", "relevance-scoring"],
            "vision": ["image-analysis", "ocr", "visual-qa"],
            "image": ["image-generation", "art-creation"],
            "audio": ["speech-recognition", "audio-analysis"],
            "video": ["video-generation", "video-analysis"]
        }
        
        # Add specific recommendations
        specific = []
        if "coder" in model_id.lower():
            specific.append("coding")
        if "instruct" in model_id.lower():
            specific.append("instruction-following")
        if "reason" in model_id.lower() or "thinking" in model_id.lower():
            specific.append("complex-reasoning")
        
        return recommendations.get(category, []) + specific
    
    def _get_vision_params(self):
        """Get vision-specific parameters"""
        return {
            "max_images": {"type": "integer", "default": 10},
            "image_detail": {"type": "string", "options": ["low", "high", "auto"], "default": "auto"},
            "image_format": {"type": "string", "options": ["url", "base64"], "default": "url"}
        }

def main():
    print("🚀 SiliconFlow Ultimate Configuration Builder")
    print("=" * 60)
    
    # Get API key
    api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    # Initialize builder
    builder = SiliconFlowConfigBuilder(api_key)
    
    # Fetch all models
    categorized_models = builder.fetch_all_models()
    
    # Print summary
    print("\n📊 Model Summary:")
    for category, models in categorized_models.items():
        if models:
            print(f"  {category.capitalize()}: {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"    - {model['id']}")
            if len(models) > 3:
                print(f"    ... and {len(models) - 3} more")
    
    # Build configurations
    print("\n⚙️ Building configurations...")
    
    crush_config = builder.build_crush_config(categorized_models)
    opencode_config = builder.build_opencode_config(categorized_models)
    
    # Create backup directory
    backup_dir = Path.home() / ".config" / "siliconflow_backup"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configurations
    crush_path = Path.home() / ".config" / "crush" / "crush.json"
    opencode_path = Path.home() / ".config" / "opencode" / "config.json"
    
    # Create directories
    crush_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing
    if crush_path.exists():
        backup_crush = backup_dir / f"crush_backup_{timestamp}.json"
        crush_path.rename(backup_crush)
        print(f"✅ Crush config backed up to: {backup_crush}")
    
    if opencode_path.exists():
        backup_opencode = backup_dir / f"opencode_backup_{timestamp}.json"
        opencode_path.rename(backup_opencode)
        print(f"✅ OpenCode config backed up to: {backup_opencode}")
    
    # Write new configs
    with open(crush_path, "w") as f:
        json.dump(crush_config, f, indent=2, ensure_ascii=False)
    
    with open(opencode_path, "w") as f:
        json.dump(opencode_config, f, indent=2, ensure_ascii=False)
    
    # Set secure permissions
    os.chmod(crush_path, 0o600)
    os.chmod(opencode_path, 0o600)
    
    print(f"\n✅ Configurations saved:")
    print(f"   Crush: {crush_path}")
    print(f"   OpenCode: {opencode_path}")
    
    # Create update script
    update_script = Path.home() / ".config" / "update_siliconflow_models.py"
    with open(update_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Auto-update SiliconFlow models
Run this periodically to keep models updated
"""
import sys
sys.path.append('.')
from siliconflow_builder import SiliconFlowConfigBuilder
import getpass
import json
from pathlib import Path

def update_models():
    # Get API key from environment or prompt
    api_key = input("Enter SiliconFlow API Key (or press Enter to use existing): ")
    if not api_key:
        # Try to extract from existing config
        try:
            with open(Path.home() / ".config" / "crush" / "crush.json", "r") as f:
                config = json.load(f)
                for provider in config.get("providers", {}).values():
                    if "api_key" in provider:
                        api_key = provider["api_key"]
                        break
        except:
            api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    builder = SiliconFlowConfigBuilder(api_key)
    categorized_models = builder.fetch_all_models()
    
    # Update configurations
    crush_config = builder.build_crush_config(categorized_models)
    opencode_config = builder.build_opencode_config(categorized_models)
    
    # Save
    with open(Path.home() / ".config" / "crush" / "crush.json", "w") as f:
        json.dump(crush_config, f, indent=2)
    
    with open(Path.home() / ".config" / "opencode" / "config.json", "w") as f:
        json.dump(opencode_config, f, indent=2)
    
    print(f"✅ Updated {len(categorized_models.get('chat', []))} chat models")
    print(f"✅ Updated {len(categorized_models.get('embedding', []))} embedding models")
    print(f"✅ Updated {len(categorized_models.get('reranker', []))} reranker models")
    print(f"✅ Updated {len(categorized_models.get('vision', []))} vision models")

if __name__ == "__main__":
    update_models()
''')
    
    os.chmod(update_script, 0o755)
    
    print(f"\n📝 Update script created: {update_script}")
    print("\n🚀 Setup Complete! Your configurations now include:")
    print("   • All available SiliconFlow models (dynamically fetched)")
    print("   • Full chat completions API with all parameters")
    print("   • Embeddings API support")
    print("   • Reranking API support")
    print("   • MCP integration for filesystem, SQLite, GitHub, web search")
    print("   • LSP integration for code completion and analysis")
    print("   • Tool calling with 128 function support")
    print("   • Reasoning traces and thinking budgets")
    print("   • Vision capabilities (if available models)")
    print("\n💡 Next steps:")
    print("   1. Restart Crush and OpenCode")
    print("   2. Run 'python ~/.config/update_siliconflow_models.py' periodically")
    print("   3. Check logs for any model-specific capabilities")

if __name__ == "__main__":
    main()
