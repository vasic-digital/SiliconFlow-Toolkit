#!/usr/bin/env python3
"""
Fixed SiliconFlow Crush Configuration Tool
Resolves provider type compatibility issues with Crush 0.22.2
"""

import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List


def fix_crush_config():
    """Fix issues with Crush configuration for SiliconFlow"""

    print("🔧 SiliconFlow Crush Configuration Fix Tool v2.0")
    print("=" * 50)

    # Load current configuration
    crush_config_path = Path.home() / ".config" / "crush" / "crush.json"

    # Check API key
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ SILICONFLOW_API_KEY environment variable not set")
        return False

    print("✅ API key found in environment")

    # Try to use dynamic model discovery first, fallback to known models
    try:
        from model_discovery import SiliconFlowModelDiscovery

        discovery = SiliconFlowModelDiscovery(api_key)
        registry = discovery.discover_all_models()

        # Use dynamic models if available
        siliconflow_models = []

        # Add chat models with proper Crush schema
        for model in registry.chat_models.models[:10]:  # Limit to 10 for performance
            siliconflow_models.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "cost_per_1m_in": model.input_pricing or 0.5,
                    "cost_per_1m_out": model.output_pricing or 0.5,
                    "cost_per_1m_in_cached": (model.input_pricing or 0.5) * 0.5,
                    "cost_per_1m_out_cached": (model.output_pricing or 0.5) * 0.5,
                    "context_window": model.context_window,
                    "default_max_tokens": model.max_tokens or 8192,
                    "can_reason": model.supports_reasoning,
                    "supports_attachments": model.supports_vision,
                    "options": {},
                }
            )

        print(f"✅ Discovered {len(siliconflow_models)} models from SiliconFlow API")

    except Exception as e:
        print(f"⚠️  Dynamic discovery failed: {e}")
        print("🔄 Using known SiliconFlow models with proper Crush schema")

        siliconflow_models = [
            {
                "id": "Qwen/Qwen2.5-14B-Instruct",
                "name": "Qwen 2.5 14B Instruct",
                "cost_per_1m_in": 0.5,
                "cost_per_1m_out": 0.5,
                "cost_per_1m_in_cached": 0.25,
                "cost_per_1m_out_cached": 0.25,
                "context_window": 32768,
                "default_max_tokens": 8192,
                "can_reason": False,
                "supports_attachments": True,
                "options": {},
            },
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen 2.5 7B Instruct",
                "cost_per_1m_in": 0.3,
                "cost_per_1m_out": 0.3,
                "cost_per_1m_in_cached": 0.15,
                "cost_per_1m_out_cached": 0.15,
                "context_window": 32768,
                "default_max_tokens": 8192,
                "can_reason": False,
                "supports_attachments": True,
                "options": {},
            },
            {
                "id": "deepseek-ai/DeepSeek-V3",
                "name": "DeepSeek V3",
                "cost_per_1m_in": 0.8,
                "cost_per_1m_out": 0.8,
                "cost_per_1m_in_cached": 0.4,
                "cost_per_1m_out_cached": 0.4,
                "context_window": 131072,
                "default_max_tokens": 8192,
                "can_reason": False,
                "supports_attachments": True,
                "options": {},
            },
            {
                "id": "deepseek-ai/DeepSeek-R1",
                "name": "DeepSeek R1",
                "cost_per_1m_in": 1.0,
                "cost_per_1m_out": 1.0,
                "cost_per_1m_in_cached": 0.5,
                "cost_per_1m_out_cached": 0.5,
                "context_window": 131072,
                "default_max_tokens": 8192,
                "can_reason": True,
                "supports_attachments": True,
                "options": {},
            },
            {
                "id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "name": "Qwen 2.5 VL 7B Instruct",
                "cost_per_1m_in": 0.6,
                "cost_per_1m_out": 0.6,
                "cost_per_1m_in_cached": 0.3,
                "cost_per_1m_out_cached": 0.3,
                "context_window": 32768,
                "default_max_tokens": 8192,
                "can_reason": False,
                "supports_attachments": True,
                "options": {},
            },
        ]

    print(f"✅ Using {len(siliconflow_models)} verified SiliconFlow models")

    # Create the corrected configuration
    config = {
        "$schema": "https://charm.land/crush.json",
        "providers": {
            "siliconflow": {
                "name": "SiliconFlow",
                "type": "openai-compat",  # Fixed: use 'openai-compat' not 'openai-compatible'
                "api_key": api_key,
                "base_url": "https://api.siliconflow.com/v1",
                "models": siliconflow_models,
            }
        },
        "defaultModel": "Qwen/Qwen2.5-14B-Instruct",
        "defaultProvider": "siliconflow",
    }

    # Backup existing config if it exists
    if crush_config_path.exists():
        backup_path = crush_config_path.with_suffix(f".backup_{int(time.time())}.json")
        try:
            shutil.copy2(crush_config_path, backup_path)
            print(f"✅ Backed up original config to: {backup_path}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")

    # Ensure config directory exists
    crush_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write fixed configuration
    try:
        with open(crush_config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Set proper permissions
        os.chmod(crush_config_path, 0o600)
        print(f"✅ Fixed configuration written to: {crush_config_path}")

    except Exception as e:
        print(f"❌ Failed to write configuration: {e}")
        return False

    # Summary
    print()
    print("📊 Configuration Fix Summary:")
    print(f"   • Total models: {len(siliconflow_models)}")
    print(f"   • Default provider: {config['defaultProvider']}")
    print(f"   • Default model: {config['defaultModel']}")
    print(f"   • Provider type: {config['providers']['siliconflow']['type']}")

    print()
    print("🎯 Next steps:")
    print("   1. Restart Crush")
    print("   2. Test with: crush run 'Hello from SiliconFlow'")
    print("   3. Check that SiliconFlow models appear in model selection")

    return True


if __name__ == "__main__":
    success = fix_crush_config()

    if success:
        print("\n🎉 Configuration fix completed successfully!")
        print("Please restart Crush to see the changes.")
    else:
        print("\n❌ Configuration fix failed.")
        print("Please check the error messages above.")
