#!/usr/bin/env python3
"""
SiliconFlow Configuration Verification Tool
Verifies that both OpenCode and Crush configurations are working properly
"""

import json
import os
from pathlib import Path


def verify_opencode_config():
    """Verify OpenCode configuration"""
    print("🔍 Verifying OpenCode Configuration...")

    config_path = Path.home() / ".config" / "opencode" / "config.json"
    if not config_path.exists():
        print("❌ OpenCode config file not found")
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check for SiliconFlow provider
        if "provider" not in config:
            print("❌ No providers found in OpenCode config")
            return False

        if "siliconflow" not in config["provider"]:
            print("❌ SiliconFlow provider not found in OpenCode config")
            return False

        siliconflow_config = config["provider"]["siliconflow"]
        required_fields = ["name", "apiKey", "baseURL", "models"]

        for field in required_fields:
            if field not in siliconflow_config:
                print(f"❌ Missing required field '{field}' in SiliconFlow provider")
                return False

        models = siliconflow_config["models"]
        if not isinstance(models, dict) or len(models) == 0:
            print("❌ No models found in SiliconFlow provider")
            return False

        print(f"✅ OpenCode config verified with {len(models)} models")
        return True

    except Exception as e:
        print(f"❌ Error verifying OpenCode config: {e}")
        return False


def verify_crush_config():
    """Verify Crush configuration"""
    print("🔍 Verifying Crush Configuration...")

    config_path = Path.home() / ".config" / "crush" / "crush.json"
    if not config_path.exists():
        print("❌ Crush config file not found")
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check for SiliconFlow provider
        if "providers" not in config:
            print("❌ No providers found in Crush config")
            return False

        if "siliconflow" not in config["providers"]:
            print("❌ SiliconFlow provider not found in Crush config")
            return False

        siliconflow_config = config["providers"]["siliconflow"]

        # Check provider type
        if siliconflow_config.get("type") != "openai-compat":
            print(f"❌ Incorrect provider type: {siliconflow_config.get('type')}")
            return False

        required_fields = ["name", "type", "api_key", "base_url", "models"]

        for field in required_fields:
            if field not in siliconflow_config:
                print(f"❌ Missing required field '{field}' in SiliconFlow provider")
                return False

        models = siliconflow_config["models"]
        if not isinstance(models, list) or len(models) == 0:
            print("❌ No models found in SiliconFlow provider")
            return False

        # Check model schema
        required_model_fields = [
            "id",
            "name",
            "cost_per_1m_in",
            "cost_per_1m_out",
            "context_window",
            "default_max_tokens",
            "can_reason",
            "supports_attachments",
            "options",
        ]

        for model in models[:3]:  # Check first 3 models
            for field in required_model_fields:
                if field not in model:
                    print(
                        f"❌ Missing required model field '{field}' in model {model.get('id', 'unknown')}"
                    )
                    return False

        print(f"✅ Crush config verified with {len(models)} models")
        return True

    except Exception as e:
        print(f"❌ Error verifying Crush config: {e}")
        return False


def verify_api_key():
    """Verify API key is available"""
    print("🔍 Verifying API Key...")

    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ SILICONFLOW_API_KEY environment variable not set")
        return False

    if not api_key.startswith("sk-"):
        print("❌ API key appears to be invalid (doesn't start with 'sk-')")
        return False

    print("✅ API key found and appears valid")
    return True


def main():
    """Main verification function"""
    print("🔧 SiliconFlow Configuration Verification Tool")
    print("=" * 50)

    # Verify API key
    api_key_ok = verify_api_key()
    print()

    # Verify OpenCode config
    opencode_ok = verify_opencode_config()
    print()

    # Verify Crush config
    crush_ok = verify_crush_config()
    print()

    # Summary
    print("📊 Verification Summary:")
    print(f"   • API Key: {'✅' if api_key_ok else '❌'}")
    print(f"   • OpenCode: {'✅' if opencode_ok else '❌'}")
    print(f"   • Crush: {'✅' if crush_ok else '❌'}")

    if api_key_ok and opencode_ok and crush_ok:
        print("\n🎉 All configurations verified successfully!")
        print("SiliconFlow should now work with both OpenCode and Crush.")
        return True
    else:
        print("\n❌ Some configurations have issues.")
        print("Please fix the problems above and run verification again.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
