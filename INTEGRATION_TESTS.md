# SiliconFlow Toolkit Integration Tests

This document provides comprehensive documentation for the integration test suite that validates the SiliconFlow Toolkit with local builds of Crush and OpenCode.

## Overview

The integration test suite performs end-to-end testing of the SiliconFlow Toolkit by:

1. **Cloning** local copies of Crush and OpenCode from GitHub
2. **Building** both applications from source
3. **Installing** SiliconFlow configurations
4. **Starting** applications in headless mode
5. **Verifying** model availability and API connectivity
6. **Testing** actual model functionality

## Prerequisites

### Required Software

- **Git** - For cloning repositories
- **Go 1.19+** - For building Crush
- **Node.js 18+** - For building OpenCode
- **npm** - For OpenCode dependencies
- **Python 3.8+** - For running tests

### Environment Variables

- **`SILICONFLOW_API_KEY`** - Required API key for SiliconFlow
  - Must be set and valid
  - Tests will fail if missing, empty, or invalid

### System Requirements

- **Disk Space**: ~2GB for cloned repositories and builds
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Internet connection for cloning and API calls

## Test Structure

```
integration_tests/
├── crush/              # Local Crush build
├── opencode/           # Local OpenCode build
└── config/             # Test configuration files
    ├── crush/
    │   └── crush.json
    └── opencode/
        └── config.json
```

## Running Tests

### Basic Usage

```bash
# Test both applications (default)
python3 integration_test.py

# Test only OpenCode
python3 integration_test.py --opencode-only

# Test only Crush
python3 integration_test.py --crush-only
```

### Custom Configuration Paths

```bash
# Use custom config home directory
python3 integration_test.py --config-home /tmp/test-config

# Use separate custom directories
python3 integration_test.py \
  --opencode-config-dir /tmp/opencode-config \
  --crush-config-dir /tmp/crush-config

# Force reinstallation
python3 integration_test.py --force
```

### Advanced Options

```bash
# Keep temporary directories after test completion
python3 integration_test.py --keep-temp

# Combine options
python3 integration_test.py \
  --opencode-only \
  --config-home /tmp/test \
  --force \
  --keep-temp
```

## Test Phases

### Phase 1: Environment Validation

**Duration**: ~10 seconds

Validates:
- ✅ `SILICONFLOW_API_KEY` environment variable
- ✅ Required tools (git, go, node, npm)
- ✅ Tool versions and functionality

**Failure Points**:
- Missing or invalid API key
- Missing required software
- Network connectivity issues

### Phase 2: Repository Cloning

**Duration**: ~2-5 minutes

Clones:
- **Crush**: `https://github.com/charmbracelet/crush.git`
- **OpenCode**: `https://github.com/sst/opencode.git`

**Failure Points**:
- Network connectivity
- Git authentication issues
- Insufficient disk space

### Phase 3: Application Building

**Duration**: ~5-15 minutes

Builds:
- **Crush**: `go build -o crush .`
- **OpenCode**: `npm install && npm run build`

**Failure Points**:
- Missing dependencies
- Build tool failures
- Compilation errors

### Phase 4: Configuration Installation

**Duration**: ~30 seconds

Runs the SiliconFlow installer with appropriate flags and custom config paths.

**Configuration Locations**:
```
Default: ~/.config/
Custom: --config-home /path/to/config
Specific: --opencode-config-dir /path --crush-config-dir /path
```

### Phase 5: Headless Application Startup

**Duration**: ~10-30 seconds

Starts applications in headless mode:
- **Crush**: `./crush --headless`
- **OpenCode**: `npm start -- --headless` (may vary)

**Environment Variables**:
- `CRUSH_CONFIG_DIR` - Custom Crush config directory
- `OPENCODE_CONFIG_DIR` - Custom OpenCode config directory

### Phase 6: Model Availability Testing

**Duration**: ~30 seconds

Validates:
- ✅ Configuration files exist and are valid JSON
- ✅ SiliconFlow providers are present in configs
- ✅ Model definitions are properly structured
- ✅ Applications are still running

### Phase 7: API Connectivity Testing

**Duration**: ~10 seconds

Tests:
- ✅ SiliconFlow API connectivity
- ✅ Model listing functionality
- ✅ Basic chat completion with test model

## Configuration Management

### Default Configuration Paths

```
OpenCode: ~/.config/opencode/config.json
Crush:    ~/.config/crush/crush.json
```

### Custom Configuration Paths

#### Global Config Home
```bash
--config-home /custom/path
# Results in:
# OpenCode: /custom/path/opencode/config.json
# Crush:    /custom/path/crush/crush.json
```

#### Application-Specific Paths
```bash
--opencode-config-dir /custom/opencode
--crush-config-dir /custom/crush
# Results in:
# OpenCode: /custom/opencode/config.json
# Crush:    /custom/crush/crush.json
```

### Configuration File Structure

#### OpenCode Configuration
```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "provider": {
    "siliconflow": {
      "name": "SiliconFlow",
      "options": {
        "apiKey": "your-key",
        "baseURL": "https://api.siliconflow.com/v1"
      },
      "models": {
        "Qwen/Qwen2.5-7B-Instruct": {
          "name": "Qwen 2.5 7B Instruct",
          "contextWindow": 32768,
          "supportsFunctionCalling": true
        }
      }
    }
  }
}
```

#### Crush Configuration
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "siliconflow-chat": {
      "name": "SiliconFlow Chat Models",
      "type": "openai-compatible",
      "api_key": "your-key",
      "base_url": "https://api.siliconflow.com/v1/chat/completions",
      "models": [
        {
          "id": "Qwen/Qwen2.5-7B-Instruct",
          "name": "Qwen 2.5 7B Instruct",
          "context_window": 32768
        }
      ]
    }
  },
  "defaultProvider": "siliconflow-chat"
}
```

## Troubleshooting

### Common Issues

#### API Key Problems
```bash
# Check if API key is set
echo $SILICONFLOW_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $SILICONFLOW_API_KEY" \
     https://api.siliconflow.com/v1/models
```

#### Build Failures

**Crush Build Issues**:
```bash
# Check Go version
go version

# Clean and rebuild
cd integration_tests/crush
go clean
go mod tidy
go build -o crush .
```

**OpenCode Build Issues**:
```bash
# Check Node.js version
node --version
npm --version

# Clean and rebuild
cd integration_tests/opencode
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### Headless Mode Issues

**Crush Headless**:
```bash
# Manual test
cd integration_tests/crush
CRUSH_CONFIG_DIR=/path/to/config ./crush --headless
```

**OpenCode Headless**:
```bash
# Check available scripts
cd integration_tests/opencode
npm run

# Manual test
OPENCODE_CONFIG_DIR=/path/to/config npm start -- --headless
```

#### Configuration Issues
```bash
# Validate JSON syntax
python3 -m json.tool config.json

# Check file permissions
ls -la config.json

# Test configuration loading
python3 -c "
import json
with open('config.json') as f:
    config = json.load(f)
print('Config is valid JSON')
"
```

### Debug Mode

Enable detailed logging:
```bash
# Set logging level
export PYTHONPATH=/path/to/siliconflow-toolkit
python3 integration_test.py 2>&1 | tee test.log

# Check logs
tail -f integration_test.log
```

### Cleanup

```bash
# Remove test directories
rm -rf integration_tests/

# Clean up Go build cache
go clean -cache -modcache

# Clean up npm cache
npm cache clean --force
```

## Test Results Interpretation

### Success Indicators

- ✅ **Environment validation passed**
- ✅ **All repositories cloned successfully**
- ✅ **Applications built without errors**
- ✅ **Configurations installed successfully**
- ✅ **Applications started in headless mode**
- ✅ **Model availability verified**
- ✅ **API connectivity confirmed**

### Performance Metrics

- **Total Duration**: ~10-25 minutes
- **Network Usage**: ~500MB (cloning)
- **Disk Usage**: ~2GB (builds + dependencies)
- **API Calls**: ~5-10 (model discovery + testing)

### Exit Codes

- **0**: All tests passed
- **1**: Tests failed
- **130**: Tests interrupted (Ctrl+C)

## Advanced Usage

### CI/CD Integration

```yaml
# .github/workflows/integration.yml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Run integration tests
        env:
          SILICONFLOW_API_KEY: ${{ secrets.SILICONFLOW_API_KEY }}
        run: python3 integration_test.py
```

### Custom Test Scenarios

```python
# Extend the test suite
from integration_test import IntegrationTestSuite

class CustomIntegrationTestSuite(IntegrationTestSuite):
    def test_custom_scenario(self):
        """Custom test scenario"""
        # Your custom test logic here
        pass

# Run custom tests
if __name__ == "__main__":
    suite = CustomIntegrationTestSuite(Path.cwd())
    suite.test_custom_scenario()
```

## Security Considerations

### API Key Handling
- API keys are read from environment variables only
- Keys are never logged or stored in files
- Test configurations use the same security practices

### Network Security
- All API calls use HTTPS
- No sensitive data is transmitted in logs
- Test repositories are cloned from official sources

### File Permissions
- Configuration files are created with 0o600 permissions
- Executable files are given 0o755 permissions
- Temporary directories are cleaned up automatically

## Contributing

### Adding New Tests

1. Extend the `IntegrationTestSuite` class
2. Add new test methods
3. Update documentation
4. Test with both applications

### Reporting Issues

When reporting test failures, include:
- Full command used
- Complete log output
- System information (OS, versions)
- API key validity status

## Conclusion

The integration test suite provides comprehensive validation of the SiliconFlow Toolkit's functionality with real applications. By testing the complete workflow from source code to running applications, it ensures that all components work together correctly and that users can successfully install and use SiliconFlow models in their preferred coding environments.