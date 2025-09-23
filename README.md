# Audio Processing Pipeline with Batch TTS System

A high-performance text-to-speech processing system built on the open-source Kokoro TTS model, featuring scalable batch processing, automated file management, and optimized audio generation pipelines. Designed for efficient audiobook production and large-scale text processing workflows.

## Technical Highlights

### Backend Engineering Features
- **Scalable Batch Processing**: Automated pipeline for processing multiple text files with progress tracking
- **Resource Management**: Dynamic CPU/GPU allocation with automatic hardware detection
- **File System Architecture**: Organized storage with automatic file lifecycle management
- **Stream Processing**: Optimized audio generation with chunked processing for long-form content
- **Error Handling**: Robust error recovery and logging throughout the processing pipeline
- **API Design**: Clean separation of concerns with modular pipeline architecture

### Infrastructure & Performance
- **Model Management**: Efficient loading and caching of 82M parameter TTS models
- **Memory Optimization**: Smart resource allocation for both CPU and GPU environments
- **Concurrent Processing**: Multi-threaded batch operations with status monitoring
- **Storage Solutions**: Automated file organization with processed content archiving
- **Progress Monitoring**: Real-time status tracking for long-running operations

## Architecture Overview

### Core Components
```
├── app.py                 # Main application with Gradio web interface
├── requirements.txt       # Python dependency management
├── packages.txt          # System-level dependencies
├── rawmaterial/          # Input text file processing queue
│   └── done/            # Processed file archive
├── output/              # Generated audio file storage
└── sample_content/      # Demo materials (Gatsby, Frankenstein)
```

### Technology Stack
- **Backend**: Python 3.10+, PyTorch, Gradio
- **AI/ML**: Kokoro-82M TTS model (82M parameters)
- **Audio Processing**: 24kHz WAV generation, NumPy audio manipulation
- **Infrastructure**: Local file system with automatic organization
- **Concurrency**: Threading for batch operations and progress tracking

## Key Engineering Contributions

### 1. Batch Processing System
- Automated discovery and processing of text files
- Queue management with priority handling
- Progress tracking with detailed status reporting
- Error recovery and partial processing capabilities

### 2. Resource Management
- Dynamic hardware detection (CPU/GPU)
- Memory-efficient model loading and caching
- Configurable processing parameters for different hardware profiles

### 3. File Pipeline Architecture
- Automated file lifecycle management
- Organized storage with archival system
- Duplicate detection and handling
- Clean separation of input, processing, and output stages

### 4. Audio Processing Optimization
- Chunked processing for long-form content
- Stream generation for real-time feedback
- Multiple voice selection with 28 available voices
- Configurable audio parameters and quality settings

## Installation & Setup

### Prerequisites
- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- 2GB storage for models and cache
- CUDA-compatible GPU (optional, but recommended)

### Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd TTS_Local
pip install -r requirements.txt

# Launch application
python app.py
# Access at http://localhost:7860
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# Windows: Download espeak-ng MSI installer
# macOS: Use conda environment for best compatibility
```

## Usage Examples

### Individual Processing
```python
# Direct API usage
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
generator = pipeline("Your text here", voice='af_heart')
```

### Batch Processing
1. Place `.txt` files in `rawmaterial/` directory
2. Use Auto-Process tab in web interface
3. Monitor progress in real-time
4. Retrieve processed audio from `output/` directory

## Performance Metrics

- **Processing Speed**: 2-5x realtime (CPU), 10-20x realtime (GPU)
- **Model Size**: 327MB core model + 523KB per voice
- **Memory Usage**: 2-4GB RAM during active processing
- **Supported Formats**: WAV output at 24kHz, 16-bit, mono
- **Voice Options**: 28 voices across American/British English variants

## Available Voices

**American English (19 voices)**  
Female: Heart, Bella, Nicole, Aoede, Kore, Sarah, Nova, Sky, Alloy, Jessica, River  
Male: Michael, Fenrir, Puck, Echo, Eric, Liam, Onyx, Santa, Adam

**British English (9 voices)**  
Female: Emma, Isabella, Alice, Lily  
Male: George, Fable, Lewis, Daniel

## Development Notes

### Built on Open Source
This project extends the [hexgrad/kokoro](https://github.com/hexgrad/kokoro) TTS model with additional batch processing capabilities and infrastructure improvements for production audiobook generation workflows.

### Key Modifications
- **Batch Processing Pipeline**: Added automated file discovery and processing
- **Progress Monitoring**: Real-time status tracking and logging
- **Resource Management**: Enhanced CPU/GPU selection and optimization
- **File Organization**: Automated archival and storage management
- **Error Handling**: Robust error recovery for long-running batch operations

## Technical Considerations

### Scalability
- Designed for processing large text collections
- Memory-efficient streaming for long-form content
- Configurable concurrency for different hardware profiles

### Security
- Local processing only - no external API dependencies after setup
- File system isolation and organized storage
- No network requirements during operation

### Monitoring
- Detailed logging for debugging and optimization
- Progress tracking for long-running operations
- Resource usage monitoring and reporting

## License

Built on the Apache 2.0 licensed Kokoro model. See original [kokoro repository](https://github.com/hexgrad/kokoro) for complete license details.