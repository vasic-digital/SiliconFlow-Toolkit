#!/usr/bin/env python3
"""
Configuration Compatibility Validator
Tests that generated configurations are compatible with Crush and OpenCode schemas
"""

import json
import os
import tempfile
from pathlib import Path
from install import EnhancedSiliconFlowConfigBuilder
from model_discovery import SiliconFlowModelDiscovery


def validate_opencode_config(config: dict) -> bool:
    """Validate OpenCode configuration structure"""
    required_fields = ["$schema", "provider"]
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False

    if "provider" in config and "siliconflow" in config["provider"]:
        sf_provider = config["provider"]["siliconflow"]
        required_provider_fields = ["name", "models"]
        for field in required_provider_fields:
            if field not in sf_provider:
                print(f"❌ Missing required provider field: {field}")
                return False

        # Check options object
        if "options" not in sf_provider:
            print("❌ Missing 'options' object in SiliconFlow provider")
            return False
        
        options = sf_provider["options"]
        if "apiKey" not in options:
            print("❌ Missing 'apiKey' in SiliconFlow options")
            return False
        if "baseURL" not in options:
            print("❌ Missing 'baseURL' in SiliconFlow options")
            return False

        # Check models structure
        models = sf_provider.get("models", {})
        if not isinstance(models, dict):
            print("❌ Models should be a dictionary")
            return False

        # Check a sample model
        if models:
            sample_model = next(iter(models.values()))
            required_model_fields = ["name", "contextWindow"]
            for field in required_model_fields:
                if field not in sample_model:
                    print(f"❌ Missing required model field: {field}")
                    return False

    print("✅ OpenCode configuration is valid")
    return True


def validate_crush_config(config: dict) -> bool:
    """Validate Crush configuration structure"""
    required_fields = ["$schema", "providers"]
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False

    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        print("❌ Providers should be a dictionary")
        return False

    # Check SiliconFlow providers
    sf_providers = [p for p in providers.keys() if "siliconflow" in p]
    if not sf_providers:
        print("❌ No SiliconFlow providers found")
        return False

    for provider_name in sf_providers:
        provider = providers[provider_name]
        required_provider_fields = ["name", "type", "api_key", "base_url", "models"]
        for field in required_provider_fields:
            if field not in provider:
                print(
                    f"❌ Missing required provider field '{field}' in {provider_name}"
                )
                return False

        # Check models structure
        models = provider.get("models", [])
        if not isinstance(models, list):
            print(f"❌ Models should be a list in {provider_name}")
            return False

        if models:
            sample_model = models[0]
            required_model_fields = ["id", "name", "context_window"]
            for field in required_model_fields:
                if field not in sample_model:
                    print(
                        f"❌ Missing required model field '{field}' in {provider_name}"
                    )
                    return False

    print("✅ Crush configuration is valid")
    return True


def test_config_generation():
    """Test configuration generation and validation"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("⚠️  No SILICONFLOW_API_KEY found, skipping real API tests")
        return

    print("🚀 Testing configuration generation and compatibility...")

    try:
        # Initialize builder
        builder = EnhancedSiliconFlowConfigBuilder(api_key)

        # Generate configurations
        print("\n📝 Generating OpenCode configuration...")
        opencode_config = builder.build_opencode_config()

        print("📝 Generating Crush configuration...")
        crush_config = builder.build_crush_config()

        # Validate configurations
        print("\n🔍 Validating configurations...")

        opencode_valid = validate_opencode_config(opencode_config)
        crush_valid = validate_crush_config(crush_config)

        if opencode_valid and crush_valid:
            print("\n✅ All configurations are valid!")

            # Save test configurations
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                opencode_file = temp_path / "opencode_test.json"
                crush_file = temp_path / "crush_test.json"

                with open(opencode_file, "w") as f:
                    json.dump(opencode_config, f, indent=2)

                with open(crush_file, "w") as f:
                    json.dump(crush_config, f, indent=2)

                print(f"📁 Test configurations saved to: {temp_path}")

                # Show summary
                registry = builder.registry
                if registry:
                    summary = builder.discovery.get_category_summary()
                    print("\n📊 Configuration Summary:")
                    print(
                        f"   OpenCode models: {len(opencode_config.get('provider', {}).get('siliconflow', {}).get('models', {}))}"
                    )
                    print(
                        f"   Crush providers: {len(crush_config.get('providers', {}))}"
                    )
                    for cat, info in summary.items():
                        if info["model_count"] > 0:
                            print(f"   {cat.title()}: {info['model_count']} models")

        else:
            print("\n❌ Configuration validation failed!")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

    return True


def test_config_merging():
    """Test configuration merging functionality"""
    print("\n🔄 Testing configuration merging...")

    builder = EnhancedSiliconFlowConfigBuilder("test-key")

    # Test OpenCode merging
    existing_opencode = {
        "theme": "dark",
        "model": "old-model",
        "provider": {"other": {"name": "Other Provider"}},
    }

    new_opencode = {
        "$schema": "https://opencode.ai/config.json",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "provider": {
            "siliconflow": {
                "name": "SiliconFlow",
                "options": {
                    "apiKey": "test-key",
                    "baseURL": "https://api.siliconflow.com/v1"
                },
                "models": {"test-model": {"name": "Test Model"}},
            }
        },
    }

    merged_opencode = builder.merge_configs_smartly(
        existing_opencode, new_opencode, "opencode"
    )

    # Verify merging preserved user settings and added new provider
    assert merged_opencode["theme"] == "dark", "Theme should be preserved"
    assert "siliconflow" in merged_opencode["provider"], (
        "SiliconFlow provider should be added"
    )
    assert "other" in merged_opencode["provider"], "Other provider should be preserved"

    # Test Crush merging
    existing_crush = {
        "defaultProvider": "old-provider",
        "providers": {"old-provider": {"name": "Old Provider"}},
    }

    new_crush = {
        "$schema": "https://charm.land/crush.json",
        "providers": {
            "siliconflow-chat": {
                "name": "SiliconFlow Chat",
                "type": "openai-compatible",
                "api_key": "test-key",
                "base_url": "https://api.siliconflow.com/v1/chat/completions",
                "models": [{"id": "test-model", "name": "Test Model"}],
            }
        },
    }

    merged_crush = builder.merge_configs_smartly(existing_crush, new_crush, "crush")

    # Verify merging
    assert "old-provider" in merged_crush["providers"], (
        "Old provider should be preserved"
    )
    assert "siliconflow-chat" in merged_crush["providers"], (
        "SiliconFlow provider should be added"
    )

    print("✅ Configuration merging tests passed!")


def test_command_line_flags():
    """Test command-line flag parsing"""
    print("\n🏁 Testing command-line flag parsing...")

    import subprocess
    import sys

    # Test --help flag
    result = subprocess.run(
        [sys.executable, "install.py", "--help"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, "Help command should succeed"
    assert "OpenCode" in result.stdout, "Help should mention OpenCode"
    assert "Crush" in result.stdout, "Help should mention Crush"

    print("✅ Command-line flag parsing tests passed!")


if __name__ == "__main__":
    success = test_config_generation()
    if success:
        test_config_merging()
        test_command_line_flags()
        print("\n🎉 All compatibility tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
