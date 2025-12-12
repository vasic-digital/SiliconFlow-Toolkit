#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for SiliconFlow Toolkit
Tests full integration with local builds of Crush and OpenCode
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Comprehensive integration test suite for SiliconFlow Toolkit"""

    def __init__(self, base_dir: Path, config_home: Optional[Path] = None):
        self.base_dir = base_dir
        self.config_home = config_home or Path.home() / ".config"
        self.test_dir = base_dir / "integration_tests"
        self.crush_dir = self.test_dir / "crush"
        self.opencode_dir = self.test_dir / "opencode"
        self.api_key = os.getenv("SILICONFLOW_API_KEY")

        # Process tracking
        self.processes = []
        self.temp_dirs = []

        # Register cleanup
        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up all processes and temporary directories"""
        logger.info("🧹 Cleaning up test environment...")

        # Terminate all processes
        for proc in self.processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                logger.warning(f"Failed to terminate process: {e}")

        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict] = None,
        timeout: int = 300,
    ) -> Tuple[int, str, str]:
        """Run a command with proper error handling"""
        try:
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            if cwd:
                logger.info(f"📁 In directory: {cwd}")

            result = subprocess.run(
                cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                logger.error(f"❌ Command failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Command timed out after {timeout} seconds")
            return -1, "", "Timeout"
        except Exception as e:
            logger.error(f"💥 Command execution failed: {e}")
            return -1, "", str(e)

    def validate_environment(self) -> bool:
        """Validate test environment and requirements"""
        logger.info("🔍 Validating test environment...")

        # Check API key
        if not self.api_key:
            logger.error("❌ SILICONFLOW_API_KEY environment variable not set")
            return False

        if not self.api_key.strip():
            logger.error("❌ SILICONFLOW_API_KEY is empty")
            return False

        logger.info("✅ SiliconFlow API key is configured")

        # Check required tools
        required_tools = ["git", "go", "node", "npm"]
        missing_tools = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            logger.error(f"❌ Missing required tools: {', '.join(missing_tools)}")
            logger.error("Please install: git, golang, nodejs, npm")
            return False

        logger.info("✅ All required tools are available")

        # Check Go version (Crush requirement)
        returncode, stdout, stderr = self.run_command(["go", "version"])
        if returncode != 0:
            logger.error("❌ Go is not working properly")
            return False

        # Extract Go version
        go_version = stdout.strip().split()[2] if stdout.strip() else "unknown"
        logger.info(f"✅ Go version: {go_version}")

        # Check Node.js version (OpenCode requirement)
        returncode, stdout, stderr = self.run_command(["node", "--version"])
        if returncode != 0:
            logger.error("❌ Node.js is not working properly")
            return False

        node_version = stdout.strip()
        logger.info(f"✅ Node.js version: {node_version}")

        return True

    def clone_repositories(self) -> bool:
        """Clone Crush and OpenCode repositories"""
        logger.info("📥 Cloning repositories...")

        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dirs.append(self.test_dir)

        # Clone Crush (Go project)
        logger.info("🐈 Cloning Crush repository...")
        returncode, stdout, stderr = self.run_command(
            [
                "git",
                "clone",
                "https://github.com/charmbracelet/crush.git",
                str(self.crush_dir),
            ]
        )

        if returncode != 0:
            logger.error("❌ Failed to clone Crush repository")
            return False

        logger.info("✅ Crush repository cloned successfully")

        # Clone OpenCode (Node.js/TypeScript project)
        logger.info("🚀 Cloning OpenCode repository...")
        returncode, stdout, stderr = self.run_command(
            [
                "git",
                "clone",
                "https://github.com/sst/opencode.git",
                str(self.opencode_dir),
            ]
        )

        if returncode != 0:
            logger.error("❌ Failed to clone OpenCode repository")
            return False

        logger.info("✅ OpenCode repository cloned successfully")

        return True

    def build_crush(self) -> bool:
        """Build Crush from source"""
        logger.info("🔨 Building Crush from source...")

        # Check if we're in the right directory
        if not (self.crush_dir / "go.mod").exists():
            logger.error("❌ Crush directory doesn't contain go.mod")
            return False

        # Build Crush
        returncode, stdout, stderr = self.run_command(
            ["go", "build", "-o", "crush", "."], cwd=self.crush_dir
        )

        if returncode != 0:
            logger.error("❌ Failed to build Crush")
            return False

        # Verify binary was created
        crush_binary = self.crush_dir / "crush"
        if not crush_binary.exists():
            logger.error("❌ Crush binary was not created")
            return False

        # Make executable
        crush_binary.chmod(0o755)
        logger.info("✅ Crush built successfully")

        return True

    def build_opencode(self) -> bool:
        """Build OpenCode from source"""
        logger.info("🔨 Building OpenCode from source...")

        # Check if we're in the right directory
        if not (self.opencode_dir / "package.json").exists():
            logger.error("❌ OpenCode directory doesn't contain package.json")
            return False

        # Install dependencies
        logger.info("📦 Installing OpenCode dependencies...")
        returncode, stdout, stderr = self.run_command(
            ["npm", "install"], cwd=self.opencode_dir
        )

        if returncode != 0:
            logger.error("❌ Failed to install OpenCode dependencies")
            return False

        # Build OpenCode
        logger.info("🏗️  Building OpenCode...")
        returncode, stdout, stderr = self.run_command(
            ["npm", "run", "build"], cwd=self.opencode_dir
        )

        if returncode != 0:
            logger.error("❌ Failed to build OpenCode")
            return False

        logger.info("✅ OpenCode built successfully")
        return True

    def install_siliconflow_configs(
        self, opencode_only: bool = False, crush_only: bool = False
    ) -> bool:
        """Install SiliconFlow configurations"""
        logger.info("⚙️  Installing SiliconFlow configurations...")

        # Set custom config home for testing
        test_config_home = self.test_dir / "config"
        test_config_home.mkdir(exist_ok=True)

        # Run install script with custom config home
        install_cmd = [sys.executable, str(self.base_dir / "install.py")]

        if opencode_only:
            install_cmd.append("--opencode")
        elif crush_only:
            install_cmd.append("--crush")
        # If neither, install both (default)

        # Set environment variables for custom config paths
        env = os.environ.copy()
        env["OPENCODE_CONFIG_DIR"] = str(test_config_home / "opencode")
        env["CRUSH_CONFIG_DIR"] = str(test_config_home / "crush")

        returncode, stdout, stderr = self.run_command(
            install_cmd, cwd=self.base_dir, env=env
        )

        if returncode != 0:
            logger.error("❌ SiliconFlow installation failed")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return False

        logger.info("✅ SiliconFlow configurations installed successfully")
        return True

    def start_crush_headless(self) -> Optional[subprocess.Popen]:
        """Start Crush in headless mode"""
        logger.info("🐈 Starting Crush in headless mode...")

        crush_binary = self.crush_dir / "crush"
        if not crush_binary.exists():
            logger.error("❌ Crush binary not found")
            return None

        # Set custom config directory
        test_config_home = self.test_dir / "config"
        env = os.environ.copy()
        env["CRUSH_CONFIG_DIR"] = str(test_config_home / "crush")

        try:
            proc = subprocess.Popen(
                [str(crush_binary), "--headless"],
                cwd=self.crush_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.processes.append(proc)

            # Wait a bit for startup
            time.sleep(3)

            # Check if process is still running
            if proc.poll() is None:
                logger.info("✅ Crush started successfully in headless mode")
                return proc
            else:
                logger.error("❌ Crush failed to start")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to start Crush: {e}")
            return None

    def start_opencode_headless(self) -> Optional[subprocess.Popen]:
        """Start OpenCode in headless mode"""
        logger.info("🚀 Starting OpenCode in headless mode...")

        # Set custom config directory
        test_config_home = self.test_dir / "config"
        env = os.environ.copy()
        env["OPENCODE_CONFIG_DIR"] = str(test_config_home / "opencode")

        try:
            # OpenCode might need different startup command
            # This may need adjustment based on actual OpenCode CLI
            proc = subprocess.Popen(
                [sys.executable, "-m", "opencode", "--headless"],
                cwd=self.opencode_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.processes.append(proc)

            # Wait a bit for startup
            time.sleep(5)

            # Check if process is still running
            if proc.poll() is None:
                logger.info("✅ OpenCode started successfully in headless mode")
                return proc
            else:
                logger.error("❌ OpenCode failed to start")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to start OpenCode: {e}")
            return None

    def test_model_availability(self, app_name: str, proc: subprocess.Popen) -> bool:
        """Test that SiliconFlow models are available in the app"""
        logger.info(f"🧪 Testing {app_name} model availability...")

        # This is a simplified test - in practice, you'd need to interact
        # with the app's API or check configuration files

        # For now, just check that the process is running and configs exist
        if proc.poll() is not None:
            logger.error(f"❌ {app_name} process is not running")
            return False

        # Check configuration files exist
        test_config_home = self.test_dir / "config"
        if app_name.lower() == "crush":
            config_file = test_config_home / "crush" / "crush.json"
        else:
            config_file = test_config_home / "opencode" / "config.json"

        if not config_file.exists():
            logger.error(f"❌ {app_name} config file not found: {config_file}")
            return False

        # Load and validate config
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if app_name.lower() == "crush":
                if "providers" not in config:
                    logger.error(f"❌ {app_name} config missing providers")
                    return False
                siliconflow_providers = [
                    p for p in config["providers"].keys() if "siliconflow" in p
                ]
                if not siliconflow_providers:
                    logger.error(f"❌ No SiliconFlow providers found in {app_name}")
                    return False
            else:  # OpenCode
                if "provider" not in config or "siliconflow" not in config["provider"]:
                    logger.error(f"❌ SiliconFlow provider not found in {app_name}")
                    return False

            logger.info(f"✅ {app_name} configuration validated")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to validate {app_name} config: {e}")
            return False

    def run_full_integration_test(
        self, opencode_only: bool = False, crush_only: bool = False
    ) -> bool:
        """Run the complete integration test suite"""
        logger.info("🚀 Starting full integration test suite...")
        logger.info("=" * 60)

        try:
            # Step 1: Environment validation
            if not self.validate_environment():
                return False

            # Step 2: Clone repositories
            if not self.clone_repositories():
                return False

            # Step 3: Build applications
            if not crush_only:
                if not self.build_crush():
                    return False

            if not opencode_only:
                if not self.build_opencode():
                    return False

            # Step 4: Install SiliconFlow configurations
            if not self.install_siliconflow_configs(opencode_only, crush_only):
                return False

            # Step 5: Start applications in headless mode
            crush_proc = None
            opencode_proc = None

            if not opencode_only:
                crush_proc = self.start_crush_headless()
                if not crush_proc:
                    return False

            if not crush_only:
                opencode_proc = self.start_opencode_headless()
                if not opencode_proc:
                    return False

            # Step 6: Test model availability
            if crush_proc and not self.test_model_availability("Crush", crush_proc):
                return False

            if opencode_proc and not self.test_model_availability(
                "OpenCode", opencode_proc
            ):
                return False

            # Step 7: Run API connectivity tests
            if not self.test_api_connectivity():
                return False

            logger.info("=" * 60)
            logger.info("🎉 All integration tests passed successfully!")
            return True

        except Exception as e:
            logger.error(f"💥 Integration test failed with exception: {e}")
            return False

    def test_api_connectivity(self) -> bool:
        """Test SiliconFlow API connectivity and model validation"""
        logger.info("🔗 Testing SiliconFlow API connectivity...")

        try:
            # Import our API client
            sys.path.insert(0, str(self.base_dir))
            from siliconflow_client import SiliconFlowAPIClient

            client = SiliconFlowAPIClient(self.api_key)

            # Test basic connectivity
            models = client.get_models()
            if not models:
                logger.error("❌ No models returned from API")
                return False

            logger.info(
                f"✅ API connectivity confirmed - {len(models)} models available"
            )

            # Test a simple chat completion
            from siliconflow_client import ChatRequest, ChatMessage

            test_request = ChatRequest(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[ChatMessage(role="user", content="Hello, test message")],
                max_tokens=10,
            )

            response = client.chat_completion(test_request)
            if "choices" not in response:
                logger.error("❌ Chat completion test failed")
                return False

            logger.info("✅ Chat completion API test passed")
            return True

        except Exception as e:
            logger.error(f"❌ API connectivity test failed: {e}")
            return False


def main():
    """Main entry point for integration tests"""
    parser = argparse.ArgumentParser(
        description="SiliconFlow Toolkit Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Integration Test Suite for SiliconFlow Toolkit

This suite performs comprehensive testing including:
- Cloning and building Crush and OpenCode from source
- Installing SiliconFlow configurations
- Starting applications in headless mode
- Verifying model availability and API connectivity

Requirements:
- SILICONFLOW_API_KEY environment variable must be set
- Git, Go, Node.js, and npm must be installed
- Sufficient disk space for cloning and building

Examples:
  %(prog)s                          # Test both Crush and OpenCode
  %(prog)s --opencode-only         # Test only OpenCode
  %(prog)s --crush-only            # Test only Crush
  %(prog)s --config-home /tmp/test # Use custom config directory
        """,
    )

    parser.add_argument(
        "--opencode-only", action="store_true", help="Test only OpenCode integration"
    )

    parser.add_argument(
        "--crush-only", action="store_true", help="Test only Crush integration"
    )

    parser.add_argument(
        "--config-home",
        type=Path,
        help="Custom config home directory (default: ~/.config)",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directories after test completion",
    )

    args = parser.parse_args()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Initialize test suite
    test_suite = IntegrationTestSuite(script_dir, args.config_home)

    if args.keep_temp:
        # Remove cleanup registration
        atexit.unregister(test_suite.cleanup)

    # Run tests
    success = test_suite.run_full_integration_test(args.opencode_only, args.crush_only)

    if success:
        print("\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Integration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
