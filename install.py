#!/usr/bin/env python3
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
                "endpoint": "/images/generations",
                "capabilities": ["image-generation"],
                "params": self._get_image_params()
            },
            "image-to-image": {
                "endpoint": "/images/edits",
                "capabilities": ["image-editing", "image-transformation"],
                "params": {}
            }
        }
    
    def _get_chat_params(self):
        return {
            "stream": {"type": "boolean", "default": True},
            "max_tokens": {"type": "integer", "default": 2048},
            "enable_thinking": {"type": "boolean", "default": False},
            "thinking_budget": {"type": "integer", "range": [128, 32768], "default": 1024},
            "min_p": {"type": "float", "range": [0, 1], "default": 0.05},
            "stop": {"type": "array", "default": []},
            "temperature": {"type": "float", "default": 0.3},
            "top_p": {"type": "float", "default": 0.9},
            "top_k": {"type": "integer", "default": 40},
            "frequency_penalty": {"type": "float", "default": 0.1},
            "n": {"type": "integer", "default": 1},
            "response_format": {"type": "object", "default": {"type": "text"}},
            "tools": {"type": "array", "max_items": 128},
            "tool_choice": {"type": "object", "default": {"type": "auto", "disable_parallel_tool_use": False}}
        }
    
    def _get_embedding_params(self):
        return {
            "encoding_format": {"type": "string", "options": ["float", "base64"], "default": "float"},
            "dimensions": {"type": "integer", "default": 768},
            "batch_size": {"type": "integer", "default": 32}
        }
    
    def _get_rerank_params(self):
        return {
            "top_n": {"type": "integer", "default": 5},
            "return_documents": {"type": "boolean", "default": True},
            "max_chunks_per_doc": {"type": "integer", "default": 5},
            "overlap_tokens": {"type": "integer", "max": 80, "default": 50}
        }
    
    def _get_image_params(self):
        return {
            "size": {"type": "string", "options": ["512x512", "1024x1024"], "default": "512x512"},
            "quality": {"type": "string", "options": ["standard", "hd"], "default": "standard"},
            "style": {"type": "string", "options": ["vivid", "natural"], "default": "natural"}
        }
    
    def fetch_all_models(self):
        all_models = []
        
        print("🔄 Fetching models from SiliconFlow API...")
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                all_models = data.get("data", [])
                print(f"✅ Found {len(all_models)} total models")
                
                if len(all_models) < 10:
                    print("⚠️  API returned limited models, using documented models")
                    all_models = self._get_documented_models()
            else:
                print(f"⚠️  API returned {response.status_code}, using documented models")
                all_models = self._get_documented_models()
                
        except Exception as e:
            print(f"⚠️  Error fetching models: {e}, using documented models")
            all_models = self._get_documented_models()
        
        categorized = self._categorize_models(all_models)
        return categorized
    
    def _get_documented_models(self):
        documented_models = []
        
        chat_models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "deepseek-ai/DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen2.5-32B-Instruct",
            "moonshotai/Kimi-K2-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "zai-org/GLM-4.6",
            "zai-org/GLM-4.5",
            "stepfun-ai/Step-2-76B",
            "baidu/ERNIE-4.5-300B-A47B",
            "tencent/Hunyuan-A13B-Instruct",
            "Qwen/Qwen3-235B-A22B-Instruct-2507"
        ]
        
        embedding_models = [
            "Qwen/Qwen3-Embedding-4B",
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-m3",
            "Alibaba-NLP/gte-Qwen2-1.5B",
            "Qwen/Qwen3-Embedding-8B"
        ]
        
        reranker_models = [
            "BAAI/bge-reranker-v2-m3",
            "Qwen/Qwen3-Reranker-4B",
            "Qwen/Qwen3-Reranker-8B",
            "Qwen/Qwen3-Reranker-0.6B"
        ]
        
        vision_models = [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-32B-Thinking",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ]
        
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
        
        for model_id in vision_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "vision"
            })
        
        return documented_models
    
    def _categorize_models(self, models):
        categorized = {
            "chat": [], "embedding": [], "reranker": [],
            "vision": [], "audio": [], "image": [], "video": []
        }
        
        for model in models:
            model_id = model.get("id", "").lower()
            
            if any(keyword in model_id for keyword in ["embedding", "bge-", "embed", "gte-"]):
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
            elif any(keyword in model_id for keyword in ["whisper", "speech", "audio", "tts", "voice"]):
                model["category"] = "audio"
                categorized["audio"].append(model)
            elif any(keyword in model_id for keyword in ["video", "sora", "gen-2"]):
                model["category"] = "video"
                categorized["video"].append(model)
            elif any(keyword in model_id for keyword in ["step-", "stepfun"]):
                model["category"] = "chat"
                categorized["chat"].append(model)
            elif "deepseek-r1" in model_id:
                model["category"] = "chat"
                categorized["chat"].append(model)
            else:
                model["category"] = "chat"
                categorized["chat"].append(model)
        
        for category in categorized:
            categorized[category] = self._sort_models_by_performance(categorized[category])
        
        return categorized
    
    def _sort_models_by_performance(self, models):
        def performance_score(model_id):
            model_id_lower = model_id.lower()
            score = 0
            
            if "0.6b" in model_id_lower or "0.6-b" in model_id_lower:
                score -= 100
            elif "1b" in model_id_lower or "1-b" in model_id_lower:
                score -= 95
            elif "1.5b" in model_id_lower or "1.5-b" in model_id_lower:
                score -= 90
            elif "2b" in model_id_lower or "2-b" in model_id_lower:
                score -= 85
            elif "4b" in model_id_lower or "4-b" in model_id_lower:
                score -= 80
            elif "7b" in model_id_lower or "7-b" in model_id_lower:
                score -= 70
            elif "8b" in model_id_lower or "8-b" in model_id_lower:
                score -= 65
            elif "14b" in model_id_lower or "14-b" in model_id_lower:
                score -= 55
            elif "32b" in model_id_lower or "32-b" in model_id_lower:
                score -= 40
            elif "72b" in model_id_lower or "72-b" in model_id_lower:
                score -= 30
            elif "76b" in model_id_lower or "76-b" in model_id_lower:
                score -= 25
            elif "235b" in model_id_lower or "235-b" in model_id_lower:
                score -= 20
            
            if "qwen2.5-7b" in model_id_lower:
                score += 5
            if "qwen2.5-14b" in model_id_lower:
                score += 8
            if "deepseek-r1" in model_id_lower:
                score += 15
            if "step-2" in model_id_lower:
                score += 10
            if "bge-m3" in model_id_lower:
                score += 8
            
            return score
        
        return sorted(models, key=lambda m: performance_score(m.get("id", "")))
    
    def build_crush_config(self, categorized_models):
        providers = {}
        
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
                        "default_max_tokens": 2048,
                        "can_reason": self._supports_reasoning(model["id"]),
                        "supports_attachments": "vl" in model["id"].lower() or "vision" in model["id"].lower(),
                        "supports_tools": True,
                        "thinking_enabled": self._supports_thinking(model["id"]),
                        "max_thinking_budget": 2048,
                        "capabilities": self._infer_capabilities(model["id"]),
                        "estimated_cost_per_1k": self._estimate_cost(model["id"]),
                        "performance_priority": idx + 1,
                        "supports_mcp": True,
                        "supports_lsp": True
                    }
                    for idx, model in enumerate(categorized_models["chat"][:15])
                ],
                "timeout": 30000,
                "retry_attempts": 2,
                "rate_limit": {"requests_per_minute": 30}
            }
        
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
                        "dimensions": 768,
                        "encoding_formats": ["float"],
                        "capabilities": ["embedding", "similarity-search"],
                        "batch_size": 32,
                        "cache_enabled": True,
                        "cache_ttl": 3600
                    }
                    for model in categorized_models["embedding"][:3]
                ],
                "timeout": 45000,
                "cache_enabled": True,
                "prefer_local_cache": True
            }
        
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
                        "max_documents": 50,
                        "return_documents": True,
                        "chunking_support": "bge-reranker" in model["id"],
                        "capabilities": ["reranking", "relevance-scoring"],
                        "default_top_n": 5,
                        "score_threshold": 0.3
                    }
                    for model in categorized_models["reranker"][:2]
                ],
                "timeout": 60000,
                "document_limit": 50
            }
        
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
                        "context_window": 32768,
                        "supports_images": True,
                        "max_image_size": "1024x1024",
                        "image_formats": ["png", "jpeg", "jpg"],
                        "capabilities": ["vision", "ocr", "image-description"],
                        "max_images_per_request": 3,
                        "image_detail": "low"
                    }
                    for model in categorized_models["vision"][:2]
                ],
                "timeout": 45000,
                "image_compression": True
            }
        
        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": "siliconflow-chat" if "siliconflow-chat" in providers else list(providers.keys())[0],
            "defaultModel": categorized_models["chat"][0]["id"] if categorized_models["chat"] else "",
            
            "sampling": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
                "enable_thinking": False,
                "thinking_budget": 1024,
                "min_p": 0.05,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.0,
                "stop_sequences": [],
                "n": 1,
                "stream": True,
                "response_format": {"type": "text"},
                "best_of": 1
            },
            
            "features": {
                "lsp_integration": {
                    "enabled": True,
                    "auto_completion": True,
                    "code_actions": True,
                    "diagnostics": True,
                    "hover": True,
                    "signature_help": True,
                    "definition": True,
                    "references": True,
                    "debounce_time": 300,
                    "max_completion_items": 20
                },
                "mcp_tools": {
                    "enabled": True,
                    "servers": [
                        {
                            "name": "filesystem",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/media/milosvasic/DATA4TB/Projects"],
                            "timeout": 15000
                        },
                        {
                            "name": "sqlite",
                            "command": "npx",
                            "args": ["-y", "@modelcontextprotocol/server-sqlite"],
                            "timeout": 10000
                        }
                    ],
                    "max_concurrent_tools": 3,
                    "tool_timeout": 10000
                },
                "session_memory": {
                    "enabled": True,
                    "max_messages": 20,
                    "compression": True
                },
                "session_caching": {
                    "enabled": True,
                    "ttl": 1800,
                    "max_size": 100
                },
                "token_usage": {
                    "enabled": True,
                    "warn_threshold": 1000,
                    "block_threshold": 4000
                },
                "reasoning_traces": {
                    "enabled": False,
                    "max_trace_length": 500
                },
                "tool_calling": {
                    "enabled": True,
                    "max_parallel_tools": 3,
                    "validation_timeout": 5000
                },
                "function_calling": {
                    "enabled": True,
                    "max_functions": 20,
                    "strict_schema": False,
                    "parallel_tool_use": True,
                    "timeout": 8000
                },
                "embeddings": {
                    "enabled": True,
                    "cache_enabled": True,
                    "batch_size": 16,
                    "precompute": False
                },
                "reranking": {
                    "enabled": True,
                    "document_limit": 50,
                    "score_threshold": 0.3
                },
                "vision": {
                    "enabled": len(categorized_models["vision"]) > 0,
                    "max_image_size": "1024x1024",
                    "compression": True
                },
                "performance": {
                    "debounce_input": 500,
                    "throttle_output": 100,
                    "lazy_loading": True,
                    "incremental_updates": True
                }
            },
            
            "model_routing": {
                "coding": "Qwen/Qwen2.5-14B-Instruct",
                "debugging": "deepseek-ai/DeepSeek-V3.2",
                "reasoning": "deepseek-ai/DeepSeek-R1",
                "complex_reasoning": "moonshotai/Kimi-K2-Instruct",
                "vision": "Qwen/Qwen2.5-VL-72B-Instruct" if categorized_models["vision"] else None,
                "chat": "Qwen/Qwen2.5-7B-Instruct",
                "quick": "Qwen/Qwen2.5-7B-Instruct",
                "embedding": "BAAI/bge-m3" if categorized_models["embedding"] else None,
                "reranking": "BAAI/bge-reranker-v2-m3" if categorized_models["reranker"] else None,
                "default": "Qwen/Qwen2.5-14B-Instruct",
                "fallback_chain": [
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-14B-Instruct",
                    "deepseek-ai/DeepSeek-V3.2",
                    "Qwen/Qwen2.5-32B-Instruct"
                ]
            },
            
            "limits": {
                "max_input_tokens": 32000,
                "max_output_tokens": 2048,
                "max_context_window": 32768,
                "max_images_per_request": 3,
                "max_tools_per_request": 5,
                "max_embedding_tokens": 8192,
                "max_rerank_documents": 50,
                "max_concurrent_requests": 3,
                "request_timeout": 30000,
                "rate_limit": {
                    "requests_per_minute": 30,
                    "tokens_per_minute": 40000
                }
            },
            
            "ui": {
                "theme": "dark",
                "show_model_info": True,
                "show_token_counts": True,
                "show_reasoning_traces": True,
                "show_tool_calls": True,
                "log_level": "info",
                "animation_enabled": True,
                "virtual_scrolling": True,
                "debounce_delay": 200
            },
            
            "agents": {
                "coder": {
                    "system_prompt": "You are an expert coding assistant. Use reasoning only for complex problems. Prioritize clean, testable code with error handling.",
                    "model": "Qwen/Qwen2.5-14B-Instruct",
                    "temperature": 0.2,
                    "enable_thinking": False,
                    "thinking_budget": 2048,
                    "max_tokens": 2048,
                    "timeout": 30000
                },
                "reasoner": {
                    "system_prompt": "You are a reasoning specialist. Break down complex problems step by step.",
                    "model": "deepseek-ai/DeepSeek-R1",
                    "temperature": 0.1,
                    "enable_thinking": True,
                    "thinking_budget": 4096,
                    "max_tokens": 4096,
                    "timeout": 60000
                },
                "debugger": {
                    "system_prompt": "Debug systematically. Analyze errors, suggest fixes. Keep responses concise.",
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "temperature": 0.1,
                    "enable_thinking": True,
                    "thinking_budget": 1024,
                    "max_tokens": 1024
                },
                "researcher": {
                    "system_prompt": "Research and analyze information. Provide concise answers with key points.",
                    "model": "moonshotai/Kimi-K2-Instruct",
                    "temperature": 0.3,
                    "max_tokens": 1536,
                    "enable_thinking": False
                },
                "vision_analyst": {
                    "system_prompt": "Analyze images concisely. Focus on key elements and actionable insights.",
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct" if any("7b" in m["id"].lower() for m in categorized_models.get("vision", [])) else 
                            (categorized_models["vision"][0]["id"] if categorized_models["vision"] else "Qwen/Qwen2.5-7B-Instruct"),
                    "temperature": 0.2,
                    "max_images": 2,
                    "image_detail": "low"
                }
            },
            
            "tools": {
                "enabled": True,
                "auto_tool_choice": True,
                "max_parallel_tools": 3,
                "timeout": 10000,
                "validation": {
                    "validate_arguments": True,
                    "allow_partial": True,
                    "strict_mode": False
                },
                "cache_results": True,
                "cache_ttl": 300
            },
            
            "embeddings": {
                "enabled": True,
                "default_model": categorized_models["embedding"][0]["id"] if categorized_models["embedding"] else None,
                "encoding_format": "float",
                "dimensions": 768,
                "normalize": True,
                "batch_size": 16,
                "cache_enabled": True,
                "cache_ttl": 3600,
                "precompute_frequent": True
            },
            
            "reranking": {
                "enabled": True,
                "default_model": categorized_models["reranker"][0]["id"] if categorized_models["reranker"] else None,
                "top_n": 5,
                "return_documents": True,
                "score_threshold": 0.3,
                "document_limit": 50,
                "chunk_size": 500
            },
            
            "options": {
                "debug": False,
                "debug_lsp": False,
                "auto_save_session": True,
                "auto_update_models": False,
                "cache_embeddings": True,
                "prefer_native_tools": True,
                "fallback_to_local": True,
                "timeout": 30000,
                "retry_attempts": 2,
                "retry_delay": 1000,
                "compression": True,
                "lazy_loading": True,
                "incremental_updates": True
            },
            
            "performance_monitoring": {
                "enabled": True,
                "sampling_rate": 0.1,
                "metrics": ["response_time", "token_usage", "cache_hit_rate"],
                "alert_thresholds": {
                    "response_time_ms": 10000,
                    "error_rate": 0.05,
                    "cache_miss_rate": 0.3
                }
            }
        }
        
        return config
    
    def build_opencode_config(self, categorized_models):
        default_chat_model = next((m for m in categorized_models.get("chat", []) 
                                 if "7b" in m["id"].lower()), 
                                categorized_models["chat"][0]["id"] if categorized_models["chat"] else "")
        
        config = {
            "$schema": "https://opencode.ai/config.json",
            "theme": "dark",
            "model": default_chat_model,
            "provider": {
                "siliconflow": {
                    "name": "SiliconFlow Full Stack",
                    "apiKey": self.api_key,
                    "baseURL": self.base_url,
                    "timeout": 30000,
                    "maxRetries": 2,
                    "retryDelay": 1000,
                    "enableStreaming": True,
                    "compression": True,
                    "defaultHeaders": {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                }
            },
            "instructions": [
                "You are a performance-optimized AI assistant. Keep responses concise and focused.",
                "Use smaller models for simple tasks, larger models only when necessary.",
                "Enable thinking mode only for complex reasoning problems.",
                "Use streaming for responses longer than 200 tokens.",
                "Cache embeddings and reuse when possible.",
                "For coding: write efficient, clean code with minimal comments.",
                "Use tools only when they provide clear value."
            ],
            "tools": {
                "fileSystem": True,
                "terminal": True,
                "git": True,
                "browser": False,
                "mcp": True,
                "embeddings": True,
                "reranking": True
            },
            "session": {
                "memoryEnabled": True,
                "maxMessages": 20,
                "autoSummarize": True,
                "summaryLength": 500,
                "embedMessages": False,
                "similarityThreshold": 0.8,
                "compression": True
            },
            "ui": {
                "showTokenCounts": True,
                "showReasoningTraces": False,
                "showToolCalls": True,
                "showEmbeddingVectors": False,
                "showRerankScores": False,
                "autoFormatCode": True,
                "theme": "dark",
                "fontSize": 13,
                "lineHeight": 1.5,
                "animationDuration": 100,
                "virtualScroll": True
            },
            "keybindings": {
                "toggleChat": "ctrl+shift+c",
                "quickCompletion": "ctrl+space",
                "explainCode": "ctrl+shift+e",
                "refactorCode": "ctrl+shift+r",
                "debugCode": "ctrl+shift+d",
                "visionAnalysis": "ctrl+shift+v"
            },
            "limits": {
                "maxContextTokens": 32768,
                "maxCompletionTokens": 2048,
                "maxEmbeddingTokens": 8192,
                "maxRerankDocuments": 50,
                "maxImagesPerRequest": 3,
                "maxToolsPerRequest": 5,
                "rateLimit": {
                    "requestsPerMinute": 30,
                    "tokensPerMinute": 40000
                },
                "timeouts": {
                    "request": 30000,
                    "tool": 10000,
                    "connection": 5000
                }
            }
        }
        
        return config
    
    def _format_model_name(self, model_id):
        parts = model_id.split("/")
        if len(parts) > 1:
            return f"{parts[0]} {parts[1]}"
        return model_id
    
    def _infer_context_window(self, model_id):
        model_id_lower = model_id.lower()
        if "128k" in model_id_lower or "200000" in model_id:
            return 131072
        elif "32k" in model_id_lower:
            return 32768
        elif "16k" in model_id_lower:
            return 16384
        elif "8k" in model_id_lower:
            return 8192
        elif "qwen3" in model_id_lower:
            return 32768
        elif "qwen2.5" in model_id_lower:
            return 16384
        else:
            return 4096
    
    def _supports_reasoning(self, model_id):
        return any(keyword in model_id.lower() for keyword in 
                  ["r1", "thinking", "k2", "reason", "deepseek-r1"])
    
    def _supports_thinking(self, model_id):
        return any(keyword in model_id.lower() for keyword in 
                  ["qwen3", "hunyuan", "glm-4.6v", "glm-4.5v", "deepseek-v3"])
    
    def _infer_capabilities(self, model_id):
        caps = []
        model_id_lower = model_id.lower()
        
        if "deepseek-r1" in model_id_lower:
            caps.append("reasoning")
            caps.append("chain-of-thought")
        if "step-2" in model_id_lower:
            caps.append("complex-analysis")
            caps.append("long-context")
        if "bge-m3" in model_id_lower:
            caps.append("multilingual")
            caps.append("dense-retrieval")
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
        if "qwen3-embedding-8b" in model_id.lower():
            return 16384
        elif "bge-large" in model_id.lower():
            return 512
        elif "bge-m3" in model_id.lower():
            return 4096
        else:
            return 2048
    
    def _estimate_cost(self, model_id):
        model_id_lower = model_id.lower()
        
        if "0.6b" in model_id_lower:
            return 0.01
        elif "1b" in model_id_lower or "1.5b" in model_id_lower:
            return 0.015
        elif "2b" in model_id_lower:
            return 0.02
        elif "4b" in model_id_lower:
            return 0.025
        elif "7b" in model_id_lower or "8b" in model_id_lower:
            return 0.035
        elif "14b" in model_id_lower:
            return 0.045
        elif "32b" in model_id_lower:
            return 0.065
        elif "72b" in model_id_lower:
            return 0.11
        elif "76b" in model_id_lower:
            return 0.12
        elif "235b" in model_id_lower:
            return 0.25
        else:
            return 0.05
    
    def _recommended_for(self, category, model_id):
        recommendations = {
            "chat": ["quick-chat", "simple-qa", "summarization"],
            "embedding": ["semantic-search", "clustering"],
            "reranker": ["document-ranking", "relevance"],
            "vision": ["image-analysis", "ocr"],
            "image": ["image-generation"],
            "audio": ["speech-recognition"],
            "video": ["video-generation"]
        }
        
        specific = []
        if "coder" in model_id.lower():
            specific.append("coding")
        if "instruct" in model_id.lower():
            specific.append("instructions")
        if "reason" in model_id.lower() or "thinking" in model_id.lower():
            specific.append("complex-reasoning")
        if "7b" in model_id.lower():
            specific.append("fast-response")
        if "deepseek-r1" in model_id.lower():
            specific.append("chain-of-thought")
        if "bge-m3" in model_id.lower():
            specific.append("multilingual-search")
        
        return recommendations.get(category, []) + specific
    
    def _get_vision_params(self):
        return {
            "max_images": {"type": "integer", "default": 2},
            "image_detail": {"type": "string", "options": ["low", "high", "auto"], "default": "low"},
            "image_format": {"type": "string", "options": ["url", "base64"], "default": "url"}
        }

def main():
    print("🚀 SiliconFlow Performance-Optimized Configuration Builder")
    print("=" * 70)
    
    api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    builder = SiliconFlowConfigBuilder(api_key)
    
    categorized_models = builder.fetch_all_models()
    
    print("\n📊 Model Summary:")
    for category, models in categorized_models.items():
        if models:
            print(f"  {category.capitalize()}: {len(models)} models")
            for model in models[:2]:
                print(f"    - {model['id']}")
    
    print("\n⚙️ Building optimized configurations...")
    
    crush_config = builder.build_crush_config(categorized_models)
    opencode_config = builder.build_opencode_config(categorized_models)
    
    backup_dir = Path.home() / ".config" / "siliconflow_backup"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    crush_path = Path.home() / ".config" / "crush" / "crush.json"
    opencode_path = Path.home() / ".config" / "opencode" / "config.json"
    
    crush_path.parent.mkdir(parents=True, exist_ok=True)
    opencode_path.parent.mkdir(parents=True, exist_ok=True)
    
    if crush_path.exists():
        backup_crush = backup_dir / f"crush_backup_{timestamp}.json"
        with open(crush_path, "r") as src, open(backup_crush, "w") as dst:
            dst.write(src.read())
        print(f"✅ Crush config backed up to: {backup_crush}")
    
    if opencode_path.exists():
        backup_opencode = backup_dir / f"opencode_backup_{timestamp}.json"
        with open(opencode_path, "r") as src, open(backup_opencode, "w") as dst:
            dst.write(src.read())
        print(f"✅ OpenCode config backed up to: {backup_opencode}")
    
    with open(crush_path, "w") as f:
        json.dump(crush_config, f, indent=2, ensure_ascii=False)
    
    with open(opencode_path, "w") as f:
        json.dump(opencode_config, f, indent=2, ensure_ascii=False)
    
    os.chmod(crush_path, 0o600)
    os.chmod(opencode_path, 0o600)
    
    print(f"\n✅ Performance-optimized configurations saved:")
    print(f"   Crush: {crush_path}")
    print(f"   OpenCode: {opencode_path}")
    
    update_script = Path.home() / ".config" / "update_siliconflow_models.py"
    with open(update_script, "w") as f:
        f.write('''#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import getpass

def extract_api_key_from_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        if "providers" in config:
            for provider in config.get("providers", {}).values():
                if isinstance(provider, dict) and "api_key" in provider:
                    return provider["api_key"]
        
        if "provider" in config:
            provider = config.get("provider", {})
            if "siliconflow" in provider:
                if "apiKey" in provider["siliconflow"]:
                    return provider["siliconflow"]["apiKey"]
                elif "options" in provider["siliconflow"]:
                    options = provider["siliconflow"]["options"]
                    if "apiKey" in options:
                        return options["apiKey"]
        
        return None
    except:
        return None

def update_models():
    print("🔄 Checking for SiliconFlow model updates...")
    
    api_key = None
    for config_path in [
        Path.home() / ".config" / "crush" / "crush.json",
        Path.home() / ".config" / "opencode" / "config.json"
    ]:
        if config_path.exists():
            api_key = extract_api_key_from_config(config_path)
            if api_key:
                print(f"✅ Found API key in {config_path.name}")
                break
    
    if not api_key:
        api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    sys.path.insert(0, str(Path.home() / ".config"))
    
    try:
        from install import SiliconFlowConfigBuilder
    except ImportError:
        print("❌ Error: install.py must be in ~/.config/ directory")
        return
    
    builder = SiliconFlowConfigBuilder(api_key)
    
    try:
        categorized_models = builder.fetch_all_models()
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")
        return
    
    try:
        crush_config = builder.build_crush_config(categorized_models)
        opencode_config = builder.build_opencode_config(categorized_models)
        
        with open(Path.home() / ".config" / "crush" / "crush.json", "w") as f:
            json.dump(crush_config, f, indent=2)
        
        with open(Path.home() / ".config" / "opencode" / "config.json", "w") as f:
            json.dump(opencode_config, f, indent=2)
        
        print(f"✅ Updated {len(categorized_models.get('chat', []))} chat models")
        print(f"✅ Updated {len(categorized_models.get('embedding', []))} embedding models")
        print(f"✅ Updated {len(categorized_models.get('reranker', []))} reranker models")
        
        print("\\n💡 Performance Tips:")
        print("  • Default model is now Qwen2.5-14B-Instruct (best balance)")
        print("  • DeepSeek-R1 for complex reasoning tasks")
        print("  • BGE-M3 for multifunction embeddings")
        print("  • Response tokens limited to 2048 by default")
        
    except Exception as e:
        print(f"❌ Failed to update configurations: {e}")

if __name__ == "__main__":
    update_models()
''')
    
    os.chmod(update_script, 0o755)
    
    perf_script = Path.home() / ".config" / "check_siliconflow_perf.py"
    with open(perf_script, "w") as f:
        f.write('''#!/usr/bin/env python3
import json
from pathlib import Path

def check_crush_config():
    config_path = Path.home() / ".config" / "crush" / "crush.json"
    if not config_path.exists():
        return ["❌ Crush config not found"]
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    optimizations = []
    
    sampling = config.get("sampling", {})
    if sampling.get("max_tokens", 0) > 4096:
        optimizations.append(f"Consider lowering max_tokens from {sampling['max_tokens']} to 2048")
    if sampling.get("temperature", 1.0) > 0.5:
        optimizations.append(f"Consider lowering temperature from {sampling.get('temperature')} to 0.3")
    if sampling.get("thinking_budget", 0) > 4096:
        optimizations.append(f"Consider lowering thinking_budget from {sampling.get('thinking_budget')} to 1024")
    
    routing = config.get("model_routing", {})
    if "14b" not in routing.get("default", "").lower():
        optimizations.append("Consider setting default model to a 14B model for best balance")
    
    if not config.get("embeddings", {}).get("cache_enabled", False):
        optimizations.append("Enable embedding cache for better performance")
    
    return optimizations if optimizations else ["✅ Crush config is performance-optimized"]

def check_opencode_config():
    config_path = Path.home() / ".config" / "opencode" / "config.json"
    if not config_path.exists():
        return ["❌ OpenCode config not found"]
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    optimizations = []
    
    if config.get("model", "") == "":
        optimizations.append("Model not set, please configure a default model")
    
    provider = config.get("provider", {}).get("siliconflow", {})
    if not provider.get("apiKey"):
        optimizations.append("API key not configured in provider")
    
    if config.get("limits", {}).get("maxCompletionTokens", 0) > 4096:
        optimizations.append("Consider lowering maxCompletionTokens to 2048")
    
    return optimizations if optimizations else ["✅ OpenCode config is performance-optimized"]

def main():
    print("🔍 SiliconFlow Performance Check")
    print("=" * 50)
    
    print("\\nChecking Crush configuration...")
    crush_optimizations = check_crush_config()
    for opt in crush_optimizations:
        print(f"  {opt}")
    
    print("\\nChecking OpenCode configuration...")
    opencode_optimizations = check_opencode_config()
    for opt in opencode_optimizations:
        print(f"  {opt}")
    
    print("\\n📊 Performance Recommendations:")
    print("  1. Use Qwen2.5-14B-Instruct for balanced performance")
    print("  2. Use DeepSeek-R1 for complex reasoning")
    print("  3. Enable streaming for responses > 200 tokens")
    print("  4. Cache embeddings with 1-hour TTL")
    print("  5. Set timeout to 30s for quick fallbacks")

if __name__ == "__main__":
    main()
''')
    
    os.chmod(perf_script, 0o755)
    
    print(f"\n📝 Performance scripts created:")
    print(f"   Update script: {update_script}")
    print(f"   Performance monitor: {perf_script}")
    
    print("\n🎯 Performance Optimizations Applied:")
    print("   1. Default model: Qwen2.5-14B-Instruct (best balance)")
    print("   2. Added DeepSeek-R1 for reasoning")
    print("   3. Added BGE-M3 embeddings (multifunction)")
    print("   4. Fixed OpenCode schema validation")
    print("   5. Added MCP/LSP support flags to Crush models")
    
    print("\n💡 Usage Tips:")
    print("   1. Run 'python ~/.config/update_siliconflow_models.py' monthly")
    print("   2. Run 'python ~/.config/check_siliconflow_perf.py' weekly")
    print("   3. Use 14B models for balanced tasks, R1 for reasoning")
    print("   4. Enable thinking mode only when needed")
    
    print("\n🔧 About MCP/LSP in Crush:")
    print("   MCP and LSP features are now explicitly enabled in model definitions")
    print("   They should appear in the model selection list")
    print("   If not visible, check that 'supports_mcp' and 'supports_lsp' are True")
    
    print("\n🚀 Restart Crush and OpenCode to apply optimizations!")

if __name__ == "__main__":
    main()
