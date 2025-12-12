#!/usr/bin/env python3
"""
Fix SiliconFlow configuration issues
Identifies and resolves problems with Crush configuration
"""

import json
import os
import time
import shutil
from pathlib import Path


def fix_crush_config():
    """Fix issues with Crush configuration that prevent models from appearing"""

    print("🔧 SiliconFlow Crush Configuration Fix Tool")
    print("=" * 50)

    # Load current configuration
    crush_config_path = Path.home() / ".config" / "crush" / "crush.json"

    if not crush_config_path.exists():
        print("❌ Crush configuration file not found")
        return False

    with open(crush_config_path, "r") as f:
        config = json.load(f)

    print(f"📁 Loaded configuration from: {crush_config_path}")

    # Check API key
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ SILICONFLOW_API_KEY environment variable not set")
        return False

    print("✅ API key found in environment")

    # Use hardcoded model list for now (bypass API discovery issues)
    print("🔄 Using known SiliconFlow models...")

    # Known good chat models
    chat_models = [
        {
            "id": "Qwen/Qwen2.5-14B-Instruct",
            "name": "Qwen 2.5 14B Instruct",
            "context_window": 32768,
            "supports_function_calling": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
        },
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen 2.5 7B Instruct",
            "context_window": 32768,
            "supports_function_calling": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
        },
        {
            "id": "deepseek-ai/DeepSeek-V3",
            "name": "DeepSeek V3",
            "context_window": 131072,
            "supports_function_calling": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
        },
        {
            "id": "deepseek-ai/DeepSeek-R1",
            "name": "DeepSeek R1",
            "context_window": 131072,
            "supports_function_calling": True,
            "supports_reasoning": True,
            "max_tokens": 8192,
        },
    ]

    # Known good vision models
    vision_models = [
        {
            "id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "name": "Qwen 2.5 VL 7B Instruct",
            "context_window": 32768,
            "supports_function_calling": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
        },
        {
            "id": "Qwen/Qwen3-VL-32B-Instruct",
            "name": "Qwen 3 VL 32B Instruct",
            "context_window": 32768,
            "supports_function_calling": True,
            "supports_reasoning": False,
            "max_tokens": 8192,
        },
    ]

    print(f"✅ Using {len(chat_models)} chat models")
    print(f"✅ Using {len(vision_models)} vision models")

    # Fix the configuration
    providers = config.get("providers", {})

    # Fix chat provider
    if chat_models:
        chat_models = registry.chat_models.models[:30]  # Limit to 30 models

        providers["siliconflow-chat"] = {
            "name": "SiliconFlow Chat Models",
            "type": "openai-compatible",
            "api_key": api_key,
            "base_url": "https://api.siliconflow.com/v1/chat/completions",
            "capabilities": [
                "chat",
                "reasoning",
                "tool-calling",
                "function-calling",
                "streaming",
            ],
            "models": [],
        }

      for model in chat_models:
            model_config = {
                "id": model["id"],
                "name": model["name"],
                "context_window": model["context_window"],
                "supports_mcp": True,
                "supports_lsp": True
            }
            
            if model["supports_function_calling"]:
                model_config["supports_function_calling"] = True
            
            if model["supports_reasoning"]:
                model_config["supports_reasoning"] = True
            
            if model.get("max_tokens"):
                model_config["max_tokens"] = model["max_tokens"]
            
            providers["siliconflow-chat"]["models"].append(model_config)

        print(f"✅ Fixed siliconflow-chat provider with {len(chat_models)} models")

    # Fix vision provider
    if registry.vision_models.models:
        vision_models = registry.vision_models.models[:15]  # Limit to 15 models

        providers["siliconflow-vision"] = {
            "name": "SiliconFlow Vision Models",
            "type": "openai-compatible",
            "api_key": api_key,
            "base_url": "https://api.siliconflow.com/v1/chat/completions",
            "capabilities": ["vision", "chat", "tool-calling", "image-understanding"],
            "models": [],
        }

        for model in vision_models:
            model_config = {
                "id": model.id,
                "name": model.name,
                "context_window": model.context_window,
                "supports_mcp": True,
                "supports_lsp": True,
                "supports_vision": True,
            }

            if model.supports_function_calling:
                model_config["supports_function_calling"] = True

            if model.max_tokens:
                model_config["max_tokens"] = model.max_tokens

            providers["siliconflow-vision"]["models"].append(model_config)

        print(f"✅ Fixed siliconflow-vision provider with {len(vision_models)} models")

    # Set default provider and model
    config["defaultProvider"] = "siliconflow-chat"

    # Use a proper chat model as default (not video model)
    default_chat_models = [
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
    ]

    chat_model_ids = [m.id for m in chat_models]
    for preferred_model in default_chat_models:
        if preferred_model in chat_model_ids:
            config["defaultModel"] = preferred_model
            print(f"✅ Set default model: {preferred_model}")
            break
    else:
        config["defaultModel"] = (
            chat_models[0].id if chat_models else "Qwen/Qwen2.5-7B-Instruct"
        )
        print(f"✅ Set default model: {config['defaultModel']}")

    # Remove or fix non-chat providers that might confuse Crush
    providers_to_remove = []
    for provider_name in providers.keys():
        if "siliconflow" in provider_name:
            base_url = providers[provider_name].get("base_url", "")
            # Remove providers that don't support chat completions
            if (
                "embeddings" in base_url
                or "rerank" in base_url
                or "audio/speech" in base_url
                or "videos/submit" in base_url
            ):
                providers_to_remove.append(provider_name)

    for provider_name in providers_to_remove:
        del providers[provider_name]
        print(f"🗑️  Removed {provider_name} provider (not chat-compatible)")

    # Update configuration
    config["providers"] = providers

    # Add metadata
    config["metadata"] = {
        "generated_by": "SiliconFlow Toolkit Fix Tool",
        "generated_at": "2024-12-12T16:47:06Z",
        "chat_providers": len(
            [
                p
                for p in providers.keys()
                if "siliconflow" in p and "chat" in providers[p].get("base_url", "")
            ]
        ),
        "total_chat_models": len(
            providers.get("siliconflow-chat", {}).get("models", [])
        ),
        "total_vision_models": len(
            providers.get("siliconflow-vision", {}).get("models", [])
        ),
    }

    # Backup existing config
    backup_path = crush_config_path.with_suffix(f".backup_{int(time.time())}.json")
    import time
    import shutil

    try:
        shutil.copy2(crush_config_path, backup_path)
        print(f"✅ Backed up original config to: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")

    # Write fixed configuration
    try:
        with open(crush_config_path, "w") as f:
            json.dump(config, f, indent=2)

        os.chmod(crush_config_path, 0o600)
        print(f"✅ Fixed configuration written to: {crush_config_path}")

    except Exception as e:
        print(f"❌ Failed to write configuration: {e}")
        return False

    # Summary
    print()
    print("📊 Configuration Fix Summary:")
    print(
        f"   • Chat models: {len(providers.get('siliconflow-chat', {}).get('models', []))}"
    )
    print(
        f"   • Vision models: {len(providers.get('siliconflow-vision', {}).get('models', []))}"
    )
    print(f"   • Default provider: {config.get('defaultProvider')}")
    print(f"   • Default model: {config.get('defaultModel')}")
    print(f"   • Total providers: {len(providers)}")

    print()
    print("🎯 Next steps:")
    print("   1. Restart Crush")
    print("   2. Check that SiliconFlow models appear in the model selection")
    print("   3. Test with a few different models")

    return True


if __name__ == "__main__":
    success = fix_crush_config()

    if success:
        print("\n🎉 Configuration fix completed successfully!")
        print("Please restart Crush to see the changes.")
    else:
        print("\n❌ Configuration fix failed.")
        print("Please check the error messages above.")
