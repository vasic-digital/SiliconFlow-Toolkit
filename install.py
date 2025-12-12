#!/usr/bin/env python3
import json
import requests
import os
import sys
from pathlib import Path
from datetime import datetime
import getpass
import shutil

class SiliconFlowConfigBuilder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.com/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
        self.default_chat_models = [
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
        
        self.default_vision_models = [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-32B-Thinking",
            "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ]
        
        self.default_audio_models = [
            "IndexTeam/IndexTTS-2",
            "fishaudio/fish-speech-1.5",
            "FunAudioLLM/CosyVoice2-0.5B"
        ]
    
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
                print(f"✅ Found {len(all_models)} total models from API")
                
                if len(all_models) < 5:
                    print("⚠️  API returned limited models, using documented models")
                    return self._get_documented_models()
                else:
                    return self._categorize_api_models(all_models)
            else:
                print(f"⚠️  API returned {response.status_code}, using documented models")
                return self._get_documented_models()
                
        except Exception as e:
            print(f"⚠️  Error fetching models: {e}, using documented models")
            return self._get_documented_models()
    
    def _categorize_api_models(self, api_models):
        categorized = {"chat": [], "vision": [], "audio": []}
        
        api_model_ids = [m.get("id", "") for m in api_models]
        
        for model_id in self.default_chat_models:
            if any(api_id == model_id for api_id in api_model_ids):
                categorized["chat"].append({"id": model_id})
        
        for model_id in self.default_vision_models:
            if any(api_id == model_id for api_id in api_model_ids):
                categorized["vision"].append({"id": model_id})
        
        for model_id in self.default_audio_models:
            if any(api_id == model_id for api_id in api_model_ids):
                categorized["audio"].append({"id": model_id})
        
        return categorized
    
    def _get_documented_models(self):
        categorized = {"chat": [], "vision": [], "audio": []}
        
        for model_id in self.default_chat_models:
            categorized["chat"].append({"id": model_id})
        
        for model_id in self.default_vision_models:
            categorized["vision"].append({"id": model_id})
        
        for model_id in self.default_audio_models:
            categorized["audio"].append({"id": model_id})
        
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
                "type": "openai-compatible",
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
                    for model in categorized_models["chat"][:20]
                ],
                "timeout": 30000,
                "retry_attempts": 2
            }
        
        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": "siliconflow-chat" if "siliconflow-chat" in providers else "",
            "defaultModel": categorized_models["chat"][0]["id"] if categorized_models["chat"] else "",
            
            "sampling": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": True
            }
        }
        
        return config

def safe_backup_and_write(config_path, new_config, backup_dir):
    config_path = Path(config_path)
    backup_dir = Path(backup_dir)
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                existing_config = json.load(f)
            
            if existing_config == new_config:
                print(f"⚠️  Config unchanged: {config_path.name}")
                return False
            
            backup_file = backup_dir / f"{config_path.stem}_backup_{timestamp}.json"
            
            with open(backup_file, "w") as f:
                json.dump(existing_config, f, indent=2)
            
            print(f"✅ Backed up existing config to: {backup_file}")
            
        except Exception as e:
            print(f"⚠️  Could not backup {config_path.name}: {e}")
            backup_file = backup_dir / f"{config_path.stem}_backup_{timestamp}_corrupted.json"
            shutil.copy2(config_path, backup_file)
            print(f"✅ Copied corrupted config to: {backup_file}")
    
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    
    os.chmod(config_path, 0o600)
    
    print(f"✅ Updated: {config_path}")
    return True

def extract_api_key_from_existing_config(config_path):
    if not Path(config_path).exists():
        return None
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        if "providers" in config:
            for provider in config.get("providers", {}).values():
                if isinstance(provider, dict) and "api_key" in provider:
                    return provider["api_key"]
        
        if "provider" in config:
            provider = config.get("provider", {})
            if "openai" in provider and "apiKey" in provider["openai"]:
                return provider["openai"]["apiKey"]
        
        return None
    except:
        return None

def main():
    print("🚀 SiliconFlow Configuration Builder")
    print("=" * 70)
    print("This script will update your Crush and OpenCode configurations.")
    print("Existing configurations will be backed up automatically.")
    print("=" * 70)
    
    backup_dir = Path.home() / ".config" / "siliconflow_backups"
    
    existing_api_key = None
    
    for config_path in [
        Path.home() / ".config" / "crush" / "crush.json",
        Path.home() / ".config" / "opencode" / "config.json"
    ]:
        key = extract_api_key_from_existing_config(config_path)
        if key:
            existing_api_key = key
            print(f"✅ Found API key in existing {config_path.name}")
            break
    
    if existing_api_key:
        use_existing = input(f"Use existing API key? (y/n): ").lower().strip()
        if use_existing == 'y':
            api_key = existing_api_key
            print("✅ Using existing API key")
        else:
            api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    else:
        api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    
    builder = SiliconFlowConfigBuilder(api_key)
    
    print("\n🔄 Fetching available models...")
    categorized_models = builder.fetch_all_models()
    
    print("\n📊 Available Models Summary:")
    for category, models in categorized_models.items():
        if models:
            print(f"  {category.upper()}: {len(models)} models")
            for model in models[:3]:
                print(f"    • {model['id']}")
            if len(models) > 3:
                print(f"    • ... and {len(models) - 3} more")
    
    print("\n⚙️  Building configurations...")
    
    crush_config = builder.build_crush_config(categorized_models)
    opencode_config = builder.build_opencode_config(categorized_models)
    
    crush_path = Path.home() / ".config" / "crush" / "crush.json"
    opencode_path = Path.home() / ".config" / "opencode" / "config.json"
    
    print("\n📁 Saving configurations...")
    
    crush_updated = safe_backup_and_write(crush_path, crush_config, backup_dir)
    opencode_updated = safe_backup_and_write(opencode_path, opencode_config, backup_dir)
    
    if not crush_updated and not opencode_updated:
        print("\n✅ No changes needed - configurations are already up to date!")
        return
    
    print("\n✅ Configuration Summary:")
    print(f"   Crush: {len(categorized_models.get('chat', []))} chat models")
    print(f"   OpenCode: {sum(len(v) for v in categorized_models.values())} total models")
    
    print("\n🔧 Setup Instructions:")
    print("   1. Set environment variables:")
    print(f"      export OPENAI_API_KEY='{api_key}'")
    print(f"      export OPENAI_BASE_URL='https://api.siliconflow.com/v1'")
    print("   2. Restart Crush and OpenCode")
    
    print("\n💡 Pro Tips:")
    print("   • Run this script again anytime to update models")
    print("   • Backups are saved to ~/.config/siliconflow_backups/")
    print("   • Use Qwen2.5-14B-Instruct for best performance/price balance")
    print("   • Use DeepSeek-R1 for complex reasoning tasks")
    
    env_file = Path.home() / ".config" / "siliconflow_env.sh"
    with open(env_file, "w") as f:
        f.write(f"""#!/bin/bash
# SiliconFlow Environment Variables
export OPENAI_API_KEY='{api_key}'
export OPENAI_BASE_URL='https://api.siliconflow.com/v1'
echo "✅ SiliconFlow environment variables set"
""")
    
    os.chmod(env_file, 0o755)
    print(f"\n📝 Environment script created: {env_file}")
    print(f"   Run: source {env_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
