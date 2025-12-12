#!/usr/bin/env python3
"""
Enhanced SiliconFlow Configuration Builder with Dynamic Model Discovery
Automatically discovers all models and capabilities from SiliconFlow API
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import getpass
import shutil
import logging
from typing import Dict, List, Optional, Any

# Import our new modules
from siliconflow_client import SiliconFlowAPIClient, SiliconFlowAPIError
from model_discovery import SiliconFlowModelDiscovery, ModelRegistry, ModelCapabilities

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedSiliconFlowConfigBuilder:
    """Enhanced configuration builder with dynamic model discovery"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.discovery = SiliconFlowModelDiscovery(api_key)
        self.registry: Optional[ModelRegistry] = None

    def initialize_registry(self) -> ModelRegistry:
        """Initialize and cache the model registry"""
        if not self.registry:
            logger.info("Initializing model registry...")
            self.registry = self.discovery.discover_all_models()
        return self.registry

    def build_opencode_config(self) -> Dict[str, Any]:
        """Build OpenCode-compatible configuration"""
        registry = self.initialize_registry()

        # Create model configurations
        models_dict = {}

        # Add chat models
        for model in registry.chat_models.models:
            models_dict[model.id] = self._create_opencode_model_config(model)

        # Add vision models
        for model in registry.vision_models.models:
            models_dict[model.id] = self._create_opencode_model_config(model)

        # Add audio models
        for model in registry.audio_models.models:
            models_dict[model.id] = self._create_opencode_model_config(model)

        # Add video models
        for model in registry.video_models.models:
            models_dict[model.id] = self._create_opencode_model_config(model)

        # Determine default model
        default_model = (
            registry.chat_models.default_model
            or registry.vision_models.default_model
            or "Qwen/Qwen2.5-14B-Instruct"  # fallback
        )

        config = {
            "$schema": "https://opencode.ai/config.json",
            "theme": "dark",
            "model": default_model,
            "provider": {
                "siliconflow": {
                    "name": "SiliconFlow",
                    "options": {
                        "apiKey": self.api_key,
                        "baseURL": "https://api.siliconflow.com/v1"
                    },
                    "models": models_dict,
                }
            },
            "instructions": ["You are a helpful AI assistant powered by SiliconFlow."],
            "tools": {
                "fileSystem": True,
                "terminal": True,
                "git": True,
                "browser": False,
                "mcp": True,
                "embeddings": True,
                "reranking": True,
            },
        }

        return config

    def build_crush_config(self) -> Dict[str, Any]:
        """Build Crush-compatible configuration"""
        registry = self.initialize_registry()

        providers = {}

        # Chat models provider
        if registry.chat_models.models:
            providers["siliconflow-chat"] = {
                "name": "SiliconFlow Chat Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": "https://api.siliconflow.com/v1/chat/completions",
                "capabilities": [
                    "chat",
                    "reasoning",
                    "tool-calling",
                    "function-calling",
                ],
                "models": [
                    self._create_crush_model_config(model)
                    for model in registry.chat_models.models[
                        :30
                    ]  # Increased to ensure enough models are visible
                ],
                "timeout": 30000,
                "retry_attempts": 2,
            }

        # Vision models provider
        if registry.vision_models.models:
            providers["siliconflow-vision"] = {
                "name": "SiliconFlow Vision Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": "https://api.siliconflow.com/v1/chat/completions",
                "capabilities": ["vision", "chat", "tool-calling"],
                "models": [
                    self._create_crush_model_config(model)
                    for model in registry.vision_models.models[:10]
                ],
                "timeout": 45000,  # Longer timeout for vision models
                "retry_attempts": 2,
            }

        # Audio models provider
        if registry.audio_models.models:
            providers["siliconflow-audio"] = {
                "name": "SiliconFlow Audio Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": "https://api.siliconflow.com/v1/audio/speech",
                "capabilities": ["audio", "speech"],
                "models": [
                    self._create_crush_model_config(model)
                    for model in registry.audio_models.models[:5]
                ],
                "timeout": 60000,  # Audio generation can take longer
                "retry_attempts": 2,
            }

        # Video models provider
        if registry.video_models.models:
            providers["siliconflow-video"] = {
                "name": "SiliconFlow Video Models",
                "type": "openai-compat",
                "api_key": self.api_key,
                "base_url": "https://api.siliconflow.com/v1/videos/submit",
                "capabilities": ["video"],
                "models": [
                    self._create_crush_model_config(model)
                    for model in registry.video_models.models[:5]
                ],
                "timeout": 120000,  # Video generation takes much longer
                "retry_attempts": 1,
            }

        # Determine default provider and model
        default_provider = (
            "siliconflow-chat"
            if "siliconflow-chat" in providers
            else list(providers.keys())[0]
            if providers
            else ""
        )
        default_model = ""
        if default_provider and default_provider in providers:
            if providers[default_provider]["models"]:
                default_model = providers[default_provider]["models"][0]["id"]

        config = {
            "$schema": "https://charm.land/crush.json",
            "providers": providers,
            "defaultProvider": default_provider,
            "defaultModel": default_model,
            "sampling": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": True,
            },
            "metadata": {
                "generated_by": "SiliconFlow Toolkit",
                "generated_at": datetime.now().isoformat(),
                "model_count": sum(
                    len(p.get("models", [])) for p in providers.values()
                ),
                "api_version": "v1",
            },
        }

        return config

    def _create_opencode_model_config(self, model: ModelCapabilities) -> Dict[str, Any]:
        """Create OpenCode model configuration"""
        config = {
            "name": model.name,
            "contextWindow": model.context_window,
            "supportsFunctionCalling": model.supports_function_calling,
            "supportsVision": model.supports_vision,
            "supportsAudio": model.supports_audio,
            "supportsVideo": model.supports_video,
        }

        if model.max_tokens:
            config["maxTokens"] = model.max_tokens

        return config

    def _create_crush_model_config(self, model: ModelCapabilities) -> Dict[str, Any]:
        """Create Crush model configuration"""
        config = {
            "id": model.id,
            "name": model.name,
            "context_window": model.context_window,
            "supports_mcp": True,
            "supports_lsp": True,
        }

        if model.max_tokens:
            config["max_tokens"] = model.max_tokens

        if model.supports_function_calling:
            config["supports_function_calling"] = True

        if model.supports_reasoning:
            config["supports_reasoning"] = True

        return config

    def merge_configs_smartly(
        self,
        existing_config: Dict[str, Any],
        new_config: Dict[str, Any],
        config_type: str,
    ) -> Dict[str, Any]:
        """Smartly merge existing config with new config"""
        if config_type == "opencode":
            return self._merge_opencode_configs(existing_config, new_config)
        elif config_type == "crush":
            return self._merge_crush_configs(existing_config, new_config)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    def _is_model_valid_opencode(self, config: Dict[str, Any], model_id: str) -> bool:
        """Check if a model ID exists in OpenCode SiliconFlow provider models"""
        try:
            models = config.get("provider", {}).get("siliconflow", {}).get("models", {})
            return model_id in models
        except (KeyError, TypeError):
            return False

    def _is_provider_valid_crush(self, config: Dict[str, Any], provider_id: str) -> bool:
        """Check if a provider ID exists in Crush providers"""
        try:
            providers = config.get("providers", {})
            return provider_id in providers
        except (KeyError, TypeError):
            return False

    def _is_model_valid_crush(self, config: Dict[str, Any], model_id: str) -> bool:
        """Check if a model ID exists in any Crush provider's models list"""
        try:
            for provider in config.get("providers", {}).values():
                for model in provider.get("models", []):
                    if model.get("id") == model_id:
                        return True
            return False
        except (KeyError, TypeError):
            return False

    def _merge_opencode_configs(
        self, existing: Dict[str, Any], new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge OpenCode configurations intelligently"""
        merged = existing.copy()

        # Preserve user settings
        user_settings = {
            "theme": existing.get("theme"),
            "instructions": existing.get("instructions"),
            "tools": existing.get("tools"),
        }

        # Update provider configuration
        if "provider" not in merged:
            merged["provider"] = {}

        # Merge SiliconFlow provider
        if "siliconflow" in new.get("provider", {}):
            merged["provider"]["siliconflow"] = new["provider"]["siliconflow"]

        # Handle model selection
        existing_model = existing.get("model")
        if existing_model and self._is_model_valid_opencode(merged, existing_model):
            # Keep existing model if it's still valid
            merged["model"] = existing_model
        else:
            # Use new model from config or fallback
            merged["model"] = new.get("model", merged.get("model"))

        # Restore other user settings
        for key, value in user_settings.items():
            if value is not None:
                merged[key] = value

        # Update schema if needed
        merged["$schema"] = new.get("$schema", merged.get("$schema"))

        return merged

    def _merge_crush_configs(
        self, existing: Dict[str, Any], new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge Crush configurations intelligently"""
        merged = existing.copy()

        # Preserve user settings (sampling only)
        user_settings = {
            "sampling": existing.get("sampling"),
        }

        # Update providers
        if "providers" not in merged:
            merged["providers"] = {}

        # Merge SiliconFlow providers
        for provider_key, provider_config in new.get("providers", {}).items():
            merged["providers"][provider_key] = provider_config

        # Handle default provider selection
        existing_provider = existing.get("defaultProvider")
        new_provider = new.get("defaultProvider")
        if new_provider and self._is_provider_valid_crush(merged, new_provider):
            # Use new default provider if provided and valid
            merged["defaultProvider"] = new_provider
        elif existing_provider and self._is_provider_valid_crush(merged, existing_provider):
            # Keep existing provider if still valid
            merged["defaultProvider"] = existing_provider
        else:
            # Fallback: use first provider or keep existing
            merged["defaultProvider"] = existing_provider or new_provider or list(merged.get("providers", {}).keys())[0]

        # Handle default model selection
        existing_model = existing.get("defaultModel")
        new_model = new.get("defaultModel")
        if new_model and self._is_model_valid_crush(merged, new_model):
            # Use new default model if provided and valid
            merged["defaultModel"] = new_model
        elif existing_model and self._is_model_valid_crush(merged, existing_model):
            # Keep existing model if still valid
            merged["defaultModel"] = existing_model
        else:
            # Fallback: use first model from default provider or keep existing
            merged["defaultModel"] = existing_model or new_model

        # Restore other user settings
        for key, value in user_settings.items():
            if value is not None:
                merged[key] = value

        # Update schema and metadata
        merged["$schema"] = new.get("$schema", merged.get("$schema"))
        merged["metadata"] = new.get("metadata", {})

        return merged


def safe_backup_and_write(
    config_path: Path, new_config: Dict[str, Any], backup_dir: Path
) -> bool:
    """Safely backup existing config and write new one"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    changed = True
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                existing_config = json.load(f)

            # Check if configs are equivalent
            if existing_config == new_config:
                logger.info(f"✅ Config unchanged: {config_path.name}")
                return False

            # Create backup
            backup_file = backup_dir / f"{config_path.stem}_backup_{timestamp}.json"
            with open(backup_file, "w") as f:
                json.dump(existing_config, f, indent=2)

            logger.info(f"✅ Backed up existing config to: {backup_file}")

        except Exception as e:
            logger.warning(f"⚠️  Could not backup {config_path.name}: {e}")
            backup_file = (
                backup_dir / f"{config_path.stem}_backup_{timestamp}_corrupted.json"
            )
            shutil.copy2(config_path, backup_file)
            logger.info(f"✅ Copied corrupted config to: {backup_file}")

    # Write new config
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    os.chmod(config_path, 0o600)
    logger.info(f"✅ Updated: {config_path}")
    return True


def extract_api_key_from_existing_config(config_path: Path) -> Optional[str]:
    """Extract API key from existing configuration"""
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check Crush format
        if "providers" in config:
            for provider in config.get("providers", {}).values():
                if isinstance(provider, dict) and "api_key" in provider:
                    return provider["api_key"]

        # Check OpenCode format
        if "provider" in config:
            provider = config.get("provider", {})
            if "siliconflow" in provider:
                sf_provider = provider["siliconflow"]
                # Check new format (options.apiKey)
                if "options" in sf_provider and "apiKey" in sf_provider["options"]:
                    return sf_provider["options"]["apiKey"]
                # Check old format (apiKey directly)
                if "apiKey" in sf_provider:
                    return sf_provider["apiKey"]

        return None
    except:
        return None


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced SiliconFlow Configuration Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Install for both OpenCode and Crush
  %(prog)s --opencode                         # Install only for OpenCode
  %(prog)s --crush                            # Install only for Crush
  %(prog)s --opencode --crush                 # Install for both (same as default)
  %(prog)s --config-home /tmp/test-config     # Use custom config home directory
  %(prog)s --opencode-config-dir /tmp/opencode # Custom OpenCode config directory
  %(prog)s --crush-config-dir /tmp/crush      # Custom Crush config directory
  %(prog)s --force                            # Force reinstallation
        """,
    )

    parser.add_argument(
        "--opencode",
        action="store_true",
        help="Install configuration for OpenCode only",
    )

    parser.add_argument(
        "--crush", action="store_true", help="Install configuration for Crush only"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force installation even if configurations exist",
    )

    parser.add_argument(
        "--config-home",
        type=Path,
        help="Custom config home directory (default: ~/.config). Both OpenCode and Crush will use subdirectories under this path.",
    )

    parser.add_argument(
        "--opencode-config-dir",
        type=Path,
        help="Custom OpenCode config directory (overrides --config-home for OpenCode)",
    )

    parser.add_argument(
        "--crush-config-dir",
        type=Path,
        help="Custom Crush config directory (overrides --config-home for Crush)",
    )

    args = parser.parse_args()

    # If neither flag is specified, install for both
    if not args.opencode and not args.crush:
        args.opencode = True
        args.crush = True

    return args


def main():
    """Main installation function"""
    args = parse_arguments()

    print("🚀 Enhanced SiliconFlow Configuration Builder")
    print("=" * 70)

    platforms = []
    if args.opencode:
        platforms.append("OpenCode")
    if args.crush:
        platforms.append("Crush")

    platform_text = " and ".join(platforms)
    print(f"This script dynamically discovers all SiliconFlow models and capabilities.")
    print(f"Installing for: {platform_text}")
    print("Existing configurations will be intelligently merged.")

    # Show custom config paths if specified
    if args.config_home:
        print(f"Config home: {args.config_home}")
    if args.opencode_config_dir:
        print(f"OpenCode config dir: {args.opencode_config_dir}")
    if args.crush_config_dir:
        print(f"Crush config dir: {args.crush_config_dir}")

    print("=" * 70)

    backup_dir = Path.home() / ".config" / "siliconflow_backups"

    existing_api_key = None

    # Try to find existing API key
    for config_path in [
        Path.home() / ".config" / "crush" / "crush.json",
        Path.home() / ".config" / "opencode" / "config.json",
    ]:
        key = extract_api_key_from_existing_config(config_path)
        if key:
            existing_api_key = key
            print(f"✅ Found API key in existing {config_path.name}")
            break

    if existing_api_key:
        use_existing = input("Use existing API key? (y/n): ").lower().strip()
        if use_existing == "y":
            api_key = existing_api_key
            print("✅ Using existing API key")
        else:
            api_key = getpass.getpass("Enter your SiliconFlow API Key: ")
    else:
        api_key = getpass.getpass("Enter your SiliconFlow API Key: ")

    try:
        # Initialize builder
        builder = EnhancedSiliconFlowConfigBuilder(api_key)

        print("\n🔄 Discovering available models...")
        registry = builder.initialize_registry()

        print("\n📊 Available Models Summary:")
        summary = builder.discovery.get_category_summary()
        for cat_name, cat_info in summary.items():
            if cat_info["model_count"] > 0:
                print(f"  {cat_name.upper()}: {cat_info['model_count']} models")
                if cat_info["default_model"]:
                    print(f"    Default: {cat_info['default_model']}")

        print("\n⚙️  Building configurations...")

        configs_to_build = []
        if args.crush:
            configs_to_build.append(("crush", builder.build_crush_config()))
        if args.opencode:
            configs_to_build.append(("opencode", builder.build_opencode_config()))

        # Determine config paths
        config_home = args.config_home or Path.home() / ".config"

        if args.crush_config_dir:
            crush_path = args.crush_config_dir / "crush.json"
        else:
            crush_path = config_home / "crush" / "crush.json"

        if args.opencode_config_dir:
            opencode_path = args.opencode_config_dir / "config.json"
        else:
            opencode_path = config_home / "opencode" / "config.json"

        print("\n📁 Saving configurations...")

        crush_updated = False
        opencode_updated = False

        for config_type, config in configs_to_build:
            if config_type == "crush":
                config_path = crush_path
                config_name = "Crush"
            else:  # opencode
                config_path = opencode_path
                config_name = "OpenCode"

            if config_path.exists() and not args.force:
                try:
                    with open(config_path, "r") as f:
                        existing_config = json.load(f)
                    merged_config = builder.merge_configs_smartly(
                        existing_config, config, config_type
                    )
                    if config_type == "crush":
                        crush_updated = safe_backup_and_write(
                            config_path, merged_config, backup_dir
                        )
                    else:
                        opencode_updated = safe_backup_and_write(
                            config_path, merged_config, backup_dir
                        )
                except Exception as e:
                    logger.warning(f"Failed to merge {config_name} config: {e}")
                    if config_type == "crush":
                        crush_updated = safe_backup_and_write(
                            config_path, config, backup_dir
                        )
                    else:
                        opencode_updated = safe_backup_and_write(
                            config_path, config, backup_dir
                        )
            else:
                if config_type == "crush":
                    crush_updated = safe_backup_and_write(
                        config_path, config, backup_dir
                    )
                else:
                    opencode_updated = safe_backup_and_write(
                        config_path, config, backup_dir
                    )

        expected_updates = args.crush + args.opencode  # Count of True values
        actual_updates = (crush_updated if args.crush else 0) + (
            opencode_updated if args.opencode else 0
        )

        if actual_updates == 0:
            print("\n✅ No changes needed - configurations are already up to date!")
            return

        print("\n✅ Configuration Summary:")
        if args.crush and crush_updated:
            crush_config = next(
                (
                    config
                    for config_type, config in configs_to_build
                    if config_type == "crush"
                ),
                None,
            )
            if crush_config:
                print(
                    f"   Crush: {crush_config.get('metadata', {}).get('model_count', 0)} models configured"
                )
        if args.opencode and opencode_updated:
            opencode_config = next(
                (
                    config
                    for config_type, config in configs_to_build
                    if config_type == "opencode"
                ),
                None,
            )
            if opencode_config:
                print(
                    f"   OpenCode: {len(opencode_config.get('provider', {}).get('siliconflow', {}).get('models', {}))} models configured"
                )

        print("\n🔧 Setup Instructions:")
        print("   1. Set environment variables:")
        print(f"      export OPENAI_API_KEY='{api_key}'")
        print(f"      export OPENAI_BASE_URL='https://api.siliconflow.com/v1'")

        restart_instructions = []
        if args.crush:
            restart_instructions.append("Crush")
        if args.opencode:
            restart_instructions.append("OpenCode")

        if restart_instructions:
            print(f"   2. Restart {' and '.join(restart_instructions)}")

        print("\n💡 Pro Tips:")
        print("   • Run this script again anytime to update models")
        print("   • Backups are saved to ~/.config/siliconflow_backups/")
        print("   • Use Qwen2.5-14B-Instruct for best performance/price balance")
        print("   • Use DeepSeek-R1 for complex reasoning tasks")

        # Create environment script
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

    except SiliconFlowAPIError as e:
        logger.error(f"SiliconFlow API Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
