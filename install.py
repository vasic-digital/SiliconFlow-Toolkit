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
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B-Instruct",
            "Qwen/Qwen3-72B-Instruct",
            "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "deepseek-ai/DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "moonshotai/Kimi-K2-Instruct",
            "zai-org/GLM-4.5",
            "zai-org/GLM-4.6",
            "stepfun-ai/Step-2-76B",
            "baidu/ERNIE-4.5-300B-A47B",
            "tencent/Hunyuan-A13B-Instruct",
            "tencent/Hunyuan-MT-7B",
            "Wan-AI/Wan2.2-I2V-A14B",
            "Wan-AI/Wan2.2-T2V-A14B"
        ]
        
        vision_models = [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-32B-Thinking",
            "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ]
        
        audio_models = [
            "IndexTeam/IndexTTS-2",
            "fishaudio/fish-speech-1.5",
            "FunAudioLLM/CosyVoice2-0.5B"
        ]
        
        for model_id in chat_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "chat"
            })
        
        for model_id in vision_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "vision"
            })
        
        for model_id in audio_models:
            documented_models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "",
                "inferred_type": "audio"
            })
        
        return documented_models
    
    def _categorize_models(self, models):
        categorized = {
            "chat": [], "vision": [], "audio": []
        }
        
        for model in models:
            model_id = model.get("id", "").lower()
            
            if any(keyword in model_id for keyword in ["vl-", "vision", "visual", "-vl"]):
                model["category"] = "vision"
                categorized["vision"].append(model)
            elif any(keyword in model_id for keyword in ["tts", "speech", "audio", "voice"]):
                model["category"] = "audio"
                categorized["audio"].append(model)
            else:
                model["category"] = "chat"
                categorized["chat"].append(model)
        
        return categorized
    
    def _format_model_name(self, model_id):
        parts = model_id.split("/")
        if len(parts) > 1:
            name = parts[1].replace("-", " ").replace("_", " ")
            return f"{parts[0]} {name}"
        return model_id
    
    def _infer_context_window(self, model_id):
        model_id_lower = model_id.lower()
        if "deepseek" in model_id_lower:
            return 131072
        elif "qwen3" in model_id_lower:
            return 32768
        elif "qwen2.5" in model_id_lower:
            return 16384
        elif "kimi" in model_id_lower or "glm" in model_id_lower:
            return 32768
        else:
            return 4096
    
    def build_opencode_config(self, categorized_models):
        models_dict = {}
        
        for category, model_list in categorized_models.items():
            for model in model_list:
                model_id = model["id"]
                model_id_lower = model_id.lower()
                
                models_dict[model_id] = {
                    "name": self._format_model_name(model_id),
                    "contextWindow": self._infer_context_window(model_id),
                    "supportsFunctionCalling": category in ["chat", "vision"],
                    "supportsVision": category == "vision",
                    "supportsAudio": category == "audio",
                    "supportsVideo": False
                }
        
        config = {
            "$schema": "https://opencode.ai/config.json",
            "theme": "dark",
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "provider": {
                "openai": {
                    "name": "SiliconFlow (OpenAI Compatible)",
                    "apiKey": self.api_key,
                    "baseURL": self.base_url,
                    "models": models_dict
                }
            },
            "instructions": [
                "You are a helpful AI assistant."
            ],
            "tools": {
                "fileSystem": True,
                "terminal": True,
                "git": True,
                "browser": False,
                "mcp": True,
                "embeddings": True,
                "reranking": True
            }
        }
        
        return config
    
    def build_crush_config(self, categorized_models):
        providers = {}
        
        if categorized_models["chat"]:
            providers["siliconflow-chat"] = {
                "name": "SiliconFlow Chat Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": f"{self.base_url}/chat/completions",
                "capabilities": ["chat", "reasoning", "tool-calling", "function-calling"],
                "models": [
                    {
                        "id": model["id"],
                        "name": self._format_model_name(model["id"]),
                        "context_window": self._infer_context_window(model["id"]),
                        "supports_mcp": True,
                        "supports_lsp": True
                    }
                    for model in categorized_models["chat"][:15]
                ],
                "timeout": 30000,
                "retry_attempts": 2
            }
        
        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": "siliconflow-chat" if "siliconflow-chat" in providers else list(providers.keys())[0],
            "defaultModel": categorized_models["chat"][0]["id"] if categorized_models["chat"] else "",
            
            "sampling": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": True
            }
        }
        
        return config

def main():
    print("🚀 SiliconFlow Configuration Builder")
    print("=" * 70)
    
    api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    builder = SiliconFlowConfigBuilder(api_key)
    
    categorized_models = builder.fetch_all_models()
    
    print("\n📊 Model Summary:")
    for category, models in categorized_models.items():
        if models:
            print(f"  {category.capitalize()}: {len(models)} models")
    
    print("\n⚙️ Building configurations...")
    
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
    
    print(f"\n✅ Configurations saved:")
    print(f"   Crush: {crush_path}")
    print(f"   OpenCode: {opencode_path}")
    
    print("\n📋 OpenCode Models Included:")
    print(f"   Chat: {len(categorized_models.get('chat', []))} models")
    print(f"   Vision: {len(categorized_models.get('vision', []))} models")
    print(f"   Audio: {len(categorized_models.get('audio', []))} models")
    
    print("\n🚀 Restart Crush and OpenCode to apply!")

if __name__ == "__main__":
    main()
    