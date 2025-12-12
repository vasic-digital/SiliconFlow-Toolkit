# AGENTS.md - Enhanced SiliconFlow Toolkit

## Essential Commands

### Installation & Setup
- **Install (both platforms)**: `./install` or `./install.sh`
- **Install (OpenCode only)**: `./install --opencode`
- **Install (Crush only)**: `./install --crush`
- **Force reinstall**: `./install --force`
- **Update tools**: `./update_tools.sh`
- **Verify setup**: `python3 ~/.config/check_siliconflow_perf.py`

### Configuration Management
- **Fix Crush config**: `python3 fix_crush_config.py`
- **Fix SiliconFlow config**: `python3 fix_siliconflow_config.py`
- **Verify configuration**: `python3 verify_config.py`
- **Sync configs**: `./sync.sh`

### Testing
- **Run unit tests**: `python3 test_siliconflow.py`
- **Compatibility tests**: `python3 test_compatibility.py`
- **Real API tests**: `SILICONFLOW_API_KEY=your-key python3 test_siliconflow.py`
- **Integration tests**: `SILICONFLOW_API_KEY=your-key python3 integration_test.py`
- **Platform-specific tests**: `python3 integration_test.py --opencode-only --crush-only`

### Development
- **Discover models**: `python3 model_discovery.py`
- **Run enhanced install**: `python3 install.py`

## Project Structure

### Core Files
- **`siliconflow_client.py`** - Main API client with all endpoints (chat, embeddings, rerank, images, audio, video)
- **`model_discovery.py`** - Dynamic model discovery system with intelligent caching
- **`install.py`** - Enhanced configuration builder for both platforms
- **`install.sh`** - Shell wrapper for installation

### Configuration Files
- **`fix_crush_config.py`** - Fixer for Crush configuration schema
- **`fix_siliconflow_config.py`** - Fixer for SiliconFlow-specific config issues
- **`verify_config.py`** - Validates configuration against both schemas
- **`sync.sh`** - Synchronizes configurations across platforms

### Test Files
- **`test_siliconflow.py`** - Unit tests for API client functionality
- **`test_compatibility.py`** - Tests for configuration schema compatibility
- **`integration_test.py`** - End-to-end integration tests requiring real API key

### Utilities
- **`update_tools.sh`** - Updates OpenCode and Crush tools via npm
- **`./Upstreams/GitHub.sh`** - Manages GitHub upstream repositories

### Documentation
- **`API_REFERENCE.md`** - Technical API documentation
- **`INTEGRATION_TESTS.md`** - Detailed integration testing guide
- **`QUICKSTART.md`** - Quick start guide for new users
- **`README.md`** - Main project documentation

## Dependencies and Environment

### Python Dependencies
- **Primary dependency**: `requests` library only (no additional packages required)
- **No package manager files**: No `requirements.txt`, `pyproject.toml`, or `setup.py` present
- **External tool dependencies**: 
  - OpenCode and Crush tools installed via npm
  - Git for repository operations
  - Go and Node.js for integration testing builds

### Runtime Requirements
- **Python**: 3.8+ with standard library only
- **API Key**: SiliconFlow API key for full functionality
- **Permissions**: Scripts require executable permissions (0o755)
- **File permissions**: Secure configs use 0o600 permissions

### Cache Management
- Model discovery uses intelligent caching with 1-hour expiry
- Cache stored in `~/.cache/siliconflow/models.json`
- Cache automatically refreshed on expiration or failure

## Code Style Guidelines

### Python Standards
- **Python version**: Python 3.8+
- **Imports**: Standard library first, then third-party (`requests`), then local imports
- **Type hints**: Use comprehensive type annotations for all functions (see `siliconflow_client.py`)
- **Dataclass pattern**: Extensive use of `@dataclass` for request/response objects (ModelCapabilities, ChatMessage, etc.)
- **Naming conventions**:
  - Functions/methods: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
  - Variables: `snake_case`
- **Path handling**: Use `pathlib.Path` instead of string paths
- **String formatting**: Use f-strings for dynamic content
- **Error handling**: Use try/except blocks with specific exception types and comprehensive logging
- **Docstrings**: Use Google-style docstrings with comprehensive descriptions

### File Organization
- **Single top-level class per file** pattern followed
- **Helper functions** grouped together under main class
- **Config validation** separated from business logic
- **Testing** uses Python `unittest` framework
- **No build system** - script-based approach instead of Makefile or CI configs

### Git and Version Control
- **.gitignore**: Comprehensive ignore patterns for temporary files, cache, and build artifacts
- **Commit messages**: Use clear, descriptive commit messages
- **Branch naming**: Use descriptive branch names (feature/, bugfix/, etc.)
- **File permissions**: Scripts should have executable permissions (0o755)
- **No CI/CD pipelines** - manual testing approach

## API Coverage

### Supported Endpoints (via `siliconflow_client.py`)
- **Chat Completions**: Full support with streaming, tools, reasoning
- **Embeddings**: Text embedding generation
- **Rerank**: Document reranking and relevance scoring
- **Images**: Text-to-image generation
- **Audio**: Speech synthesis and transcription
- **Video**: Text-to-video generation
- **Models**: Dynamic model discovery and metadata

### Model Categories
- **Chat Models**: 77+ models for conversational AI
- **Vision Models**: 12+ models for image understanding
- **Audio Models**: 3+ models for speech synthesis
- **Video Models**: 2+ models for video generation

## Configuration Compatibility

### OpenCode Schema
- Fully compatible with `https://opencode.ai/config.json`
- Supports all provider and model configuration options
- Preserves user settings during merges
- Intelligent default model selection based on capabilities
- **Timeout & Retry Configuration**: Includes `timeout: 30000` and `maxRetries: 2` to prevent hanging requests
- **Chat-Only Filtering**: Only chat-capable models are exposed to OpenCode (non-chat models filtered out)

### Crush Schema
- Fully compatible with `https://charm.land/crush.json`
- Multiple provider support (chat, vision, audio, video)
- Intelligent default model selection
- Maintains backward compatibility with existing configurations

## Development Workflow

### Standard Process
1. **Model Discovery**: Automatically fetch and categorize all SiliconFlow models
2. **Configuration Generation**: Create compatible configs for both platforms
3. **Smart Merging**: Update existing configs without breaking changes
4. **Validation**: Ensure schema compliance and functionality
5. **Testing**: Comprehensive test suite with real API integration
6. **Caching**: Intelligent response caching for performance

### Code Change Process
1. **Start with unit tests**: Modify/add tests in `test_siliconflow.py`
2. **Update API client**: Modify `siliconflow_client.py` with type hints and docstrings
3. **Update configuration**: Modify schema handling in `install.py` and related files
4. **Run compatibility tests**: Use `python3 test_compatibility.py`
5. **Integration test**: Run with API key if available
6. **Update documentation**: Update relevant .md files

## Testing Patterns

### Unit Tests (`test_siliconflow.py`)
- Uses Python `unittest` framework
- Mock-based testing of API responses
- Tests all endpoints with mocked data
- Configuration schema validation tests

### Compatibility Tests (`test_compatibility.py`)
- Validates schema compatibility between platforms
- Tests configuration merging logic
- Ensures backward compatibility

### Integration Tests (`integration_test.py`)
- **Requires real API key**: Set `SILICONFLOW_API_KEY` environment variable
- Clones and builds: 
  - Crush from source (requires Go)
  - OpenCode from source (requires Node.js)
- Tests with real SiliconFlow API
- Validates model availability and functionality
- Headless operation testing

### Test Command Patterns
```bash
# Unit tests (no API key needed)
python3 test_siliconflow.py

# Compatibility tests
python3 test_compatibility.py

# Real API tests (requires key)
SILICONFLOW_API_KEY=your-key python3 test_siliconflow.py

# Integration tests (requires key and build tools)
SILICONFLOW_API_KEY=your-key python3 integration_test.py

# Platform-specific integration tests
SILICONFLOW_API_KEY=your-key python3 integration_test.py --opencode-only
SILICONFLOW_API_KEY=your-key python3 integration_test.py --crush-only
```

## Important Gotchas and Patterns

### Security Patterns
- **API keys**: Never hardcode; use environment variables or secure configs
- **File permissions**: Secure configs use 0o600 permissions
- **No secrets in logs**: API keys and sensitive data are never logged
- **Backup creation**: Always creates backups before modifying existing configs

### Performance Optimizations
- **Caching**: Model discovery uses 1-hour cache to reduce API calls
- **Batch processing**: Where possible, uses batch operations
- **Connection pooling**: Uses `requests.Session` for connection reuse

### Error Handling Patterns
- **Graceful degradation**: Falls back to cached data on API failures
- **Retry logic**: Implements intelligent retry with exponential backoff
- **Comprehensive logging**: Uses Python logging with appropriate levels
- **Input validation**: Validates all inputs before API calls

### Configuration Merging Rules
- Always preserve user settings when merging configurations
- Intelligent default model selection based on capabilities
- Schema validation before writing any config files
- Backup existing configs before modifications

### Platform-Specific Considerations
- **OpenCode**: Uses npm for tool installation and updates
- **Crush**: Uses Go modules for builds
- **SiliconFlow**: Direct API integration with comprehensive error handling

## Linting and Code Quality

### No Formal Linters
- No `ruff`, `black`, `flake8`, or other formatters configured
- No CI/CD pipelines enforcing code style
- Manual code review approach

### Style Enforcement
- Follow existing code patterns in `siliconflow_client.py`
- Use comprehensive type hints for all functions
- Use Google-style docstrings
- Maintain consistent indentation (4 spaces)

## Useful Commands for Development

### Model Discovery
```bash
# Run model discovery (populates cache)
python3 model_discovery.py

# Force refresh cache
python3 model_discovery.py --force
```

### Configuration Testing
```bash
# Test configuration without API key
python3 test_compatibility.py

# Verify a specific config file
python3 verify_config.py path/to/config.json
```

### System Verification
```bash
# Check system setup
python3 ~/.config/check_siliconflow_perf.py

# Update all tools
./update_tools.sh
```

### Git Operations (Project Specific)
```bash
# Manage GitHub upstreams
./Upstreams/GitHub.sh

# See README.pdf for visual documentation
```

## Project Architecture Notes

### Single Responsibility Principle
- Each file has a clear, single purpose
- `siliconflow_client.py`: API interactions only
- `model_discovery.py`: Model metadata management only
- `install.py`: Configuration building only

### No Framework Dependencies
- Pure Python with only `requests` dependency
- No web frameworks, ORMs, or complex dependencies
- Script-based instead of application-based

### Configuration-Driven Design
- All platform integrations are configuration-based
- Schema-driven compatibility validation
- Extensible model discovery system

### Testing Philosophy
- Mock-based unit tests for fast iteration
- Real API integration tests for confidence
- Configuration schema tests for compatibility
- Headless testing for OpenCode/Crush integration