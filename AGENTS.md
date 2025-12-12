# AGENTS.md - Enhanced SiliconFlow Toolkit

## Build/Lint/Test Commands

### Installation & Setup
- **Install (both platforms)**: `./install` or `./install.sh`
- **Install (OpenCode only)**: `./install --opencode`
- **Install (Crush only)**: `./install --crush`
- **Force reinstall**: `./install --force`
- **Update tools**: `./update_tools.sh`
- **Verify setup**: `python3 ~/.config/check_siliconflow_perf.py`

### Configuration Management
- **Fix Crush config**: `python3 fix_crush_config.py`
- **Sync configs**: `./sync.sh`

### Testing
- **Run unit tests**: `python3 test_siliconflow.py`
- **Compatibility tests**: `python3 test_compatibility.py`
- **Real API tests**: `SILICONFLOW_API_KEY=your-key python3 test_siliconflow.py`
- **Integration tests**: `SILICONFLOW_API_KEY=your-key python3 integration_test.py`
- **Platform-specific tests**: `python3 integration_test.py --opencode-only --crush-only`

## Code Style Guidelines

### Python Standards
- **Python version**: Python 3.8+
- **Imports**: Standard library first, then third-party, then local imports
- **Type hints**: Use comprehensive type annotations for all functions
- **Naming conventions**:
  - Functions/methods: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
  - Variables: `snake_case`
- **Path handling**: Use `pathlib.Path` instead of string paths
- **String formatting**: Use f-strings for dynamic content
- **Error handling**: Use try/except blocks with specific exception types
- **Docstrings**: Use Google-style docstrings with comprehensive descriptions

### Git and Version Control
- **.gitignore**: Comprehensive ignore patterns for temporary files, cache, and build artifacts
- **Commit messages**: Use clear, descriptive commit messages
- **Branch naming**: Use descriptive branch names (feature/, bugfix/, etc.)
- **File permissions**: Scripts should have executable permissions (0o755)

### Code Structure
- **Architecture**: Class-based with clear separation of concerns
- **File organization**: One primary class per file with helper functions
- **Configuration**: JSON-based with schema validation
- **Security**: Store API keys securely with 0o600 permissions
- **Caching**: Implement intelligent response caching (1-hour expiry)
- **Logging**: Comprehensive logging with appropriate levels

### Best Practices
- **Backup configs**: Always create backups before modifying existing files
- **Input validation**: Validate API keys and configuration data
- **Error recovery**: Graceful degradation with fallback mechanisms
- **Performance**: Implement caching and batch processing
- **Testing**: Comprehensive test coverage with real API integration
- **Documentation**: Detailed docstrings and usage examples
- **Modularity**: Break complex operations into smaller, testable functions

## API Coverage

### Supported Endpoints
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

### Crush Schema
- Fully compatible with `https://charm.land/crush.json`
- Multiple provider support (chat, vision, audio, video)
- Intelligent default model selection

## Development Workflow

1. **Model Discovery**: Automatically fetch and categorize all SiliconFlow models
2. **Configuration Generation**: Create compatible configs for both platforms
3. **Smart Merging**: Update existing configs without breaking changes
4. **Validation**: Ensure schema compliance and functionality
5. **Testing**: Comprehensive test suite with real API integration
6. **Caching**: Intelligent response caching for performance</content>
<parameter name="filePath">/media/milosvasic/DATA4TB/Projects/SuperAgent/Toolkit/SiliconFlow/AGENTS.md