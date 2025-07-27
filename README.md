

# Sonific - Advanced 3D Data Sonification Platform

**Transform your 3D datasets into intelligent audio experiences through machine learning and spatial audio**

Sonific is a cutting-edge Python application that converts complex 3D spatial data (DEMs, LiDAR, point clouds) into meaningful, interactive audio representations. Using advanced machine learning algorithms and sophisticated audio synthesis, Sonific makes 3D data accessible through sound, perfect for research, accessibility applications, and data exploration.

## 🎵 Key Features

- **Multiple File Format Support**: TIFF/DEM, LiDAR LAS, PLY point clouds, CSV/TXT coordinate data
- **Advanced ML Processing**: DBSCAN/K-means clustering, anomaly detection, surface classification
- **Three Distinct Sonification Modes**: Clean (scientific), Spatial (geographic), Advanced (multi-track)
- **Professional Audio Synthesis**: 7 different instrument types with realistic ADSR envelopes
- **3D Spatial Audio**: HRTF processing for accurate spatial positioning
- **AI-Powered Analysis**: LLM-based audio quality assessment and interpretation guidance
- **Interactive Web Interface**: Professional Streamlit-based UI with real-time processing
- **Large Dataset Support**: Handles files up to 10GB+ with intelligent optimization
- **Multi-format Output**: WAV audio files and MIDI for musical analysis

## 🚀 Installation

### Prerequisites

- **Python 3.8-3.11** (Open3D compatibility requirement)
- **16GB+ RAM** (recommended for large datasets)
- **Audio output device** (headphones recommended for spatial audio)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/sonific.git
cd sonific
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv sonific_env

# Activate virtual environment
# On Windows:
sonific_env\Scripts\activate
# On macOS/Linux:
source sonific_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test import of core modules
python -c "import streamlit, numpy, open3d, librosa; print('✅ Installation successful!')"
```

## 📋 Requirements.txt

```txt
# Core Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# 3D Data Processing
open3d>=0.17.0
tifffile>=2023.7.0
laspy>=2.4.0

# Audio Processing & Synthesis
librosa>=0.10.0
soundfile>=0.12.0
midiutil>=1.2.1

# Machine Learning & AI
langchain-openai>=0.1.0
langchain-core>=0.2.0
langchain-community>=0.2.0

# Web Application
streamlit>=1.28.0
plotly>=5.15.0
```

## 🎮 Quick Start

### Basic Usage

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload your data**: 
   - Support formats: TIFF, LAS, PLY, CSV
   - File size: Up to 10GB+
   - Coordinates: Any coordinate system

4. **Configure settings**:
   - **Quality vs Speed**: Lower = better quality, Higher = faster
   - **Max Points**: More points = more detail
   - **Advanced ML Features**: Enable for best results

5. **Analyze your data**:
   - Click "ANALYZE 3D DATA"
   - Review clustering and anomaly detection results
   - Examine 3D visualization

6. **Generate sonification**:
   - Choose sonification mode
   - Set tempo (60-180 BPM) and duration (10-60 seconds)
   - Click "GENERATE AUDIO FILES"

7. **Listen and download**:
   - Play audio directly in browser
   - Download WAV and MIDI files
   - Get AI analysis of audio quality

## 🎵 Sonification Modes

### 1. Clean Mode
- **Purpose**: Scientific-grade continuous audio representation
- **Output**: Smooth drone sounds that follow elevation changes
- **Best for**: Precise elevation analysis, accessibility applications
- **Audio characteristics**: Continuous tones, smooth transitions, no clicking

### 2. Spatial Mode  
- **Purpose**: Geographic sweep with distinct cluster sounds
- **Output**: Rhythmic patterns with different instruments per terrain type
- **Best for**: Understanding spatial relationships, terrain classification
- **Audio characteristics**: Beat-based patterns, instrument variety, spatial movement

### 3. Advanced Mode
- **Purpose**: Professional multi-track rendering using all ML features
- **Output**: Layered audio with background + clusters + features + anomalies
- **Best for**: Research publications, comprehensive analysis
- **Audio characteristics**: Rich, layered, publication-quality audio

Here's the updated README section for the AI Assistant setup, removing references to LM Studio and focusing on OpenAI-compatible endpoints:

## 🤖 AI Assistant Setup (Optional)

The AI assistant provides expert analysis of your generated sonification files and helps interpret audio quality. This feature is **completely optional** - the platform works perfectly without it.

### OpenAI-Compatible Endpoint Configuration

To enable AI-powered audio analysis, you'll need access to any OpenAI-compatible API endpoint. This could be:

- **OpenAI API** (gpt-3.5-turbo, gpt-4, etc.)
- **Azure OpenAI Service**
- **Local AI servers** (Ollama, vLLM, etc.)
- **Other OpenAI-compatible providers**

### Configuration Steps

1. **Get your API credentials**:
   - OpenAI API key from https://platform.openai.com/
   - Or credentials from your preferred provider

2. **Configure the AI agent**:
   - Open `ai_agent.py` in your project
   - Update the connection settings in the `setup_llm()` method:

```python
def setup_llm(self):
    """Initialize AI connection with your OpenAI-compatible endpoint"""
    try:
        self.llm = ChatOpenAI(
            base_url="https://api.openai.com/v1",  # Your API endpoint
            api_key="your-api-key-here",          # Your API key
            model="gpt-3.5-turbo",                # Your preferred model
            temperature=0.7,
            max_tokens=2000,
            timeout=60
        )
        # Test connection
        test_response = self.llm.invoke([HumanMessage(content="Hello")])
        print("✅ AI Assistant connection successful")
        self.llm_available = True
    except Exception as e:
        print(f"⚠️ AI Assistant connection failed: {str(e)}")
        print("AI Agent will run in offline mode with expert responses")
        self.llm = None
        self.llm_available = False
```

3. **Alternative: Environment Variables**
   Create a `.env` file in your project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-3.5-turbo
   ```

### What the AI Assistant Provides

- **Audio Quality Analysis**: Technical assessment of your generated sonification
- **Interpretation Guidance**: Explains what specific sounds mean in your data
- **Parameter Recommendations**: Suggests settings improvements based on your dataset
- **Scientific Context**: Helps describe sonification results for research papers

### Offline Mode (No Setup Required)

If you don't configure an AI endpoint, the assistant automatically runs in **offline mode** with:
- **Expert pre-built responses** for common sonification questions
- **Technical audio analysis** using built-in algorithms
- **Parameter guidance** based on dataset characteristics
- **Full functionality** without requiring external AI services

### Privacy Note

- **Offline mode**: No data leaves your computer
- **Online mode**: Only audio analysis metadata (not raw data) is sent to your configured AI service
- **Your choice**: The platform is designed to work excellently either way

The AI assistant enhances the experience but is not required for the core sonification functionality. All audio generation, processing, and analysis can be performed entirely offline.



## 📊 Supported Data Formats

| Format | Extension | Description | Max Size | Use Case |
|--------|-----------|-------------|----------|----------|
| **DEM TIFF** | `.tif`, `.tiff` | Digital elevation models | 10GB+ | Terrain mapping, topography |
| **LiDAR LAS** | `.las` | Laser scan point clouds | 2GB | Building scanning, forestry |
| **Point Cloud PLY** | `.ply` | 3D point cloud data | 1GB | 3D models, archaeological data |
| **CSV/TXT** | `.csv`, `.txt` | X,Y,Z coordinate tables | 500MB | Custom datasets, survey data |

### Data Requirements
- **Minimum points**: 100 (for meaningful analysis)
- **Coordinates**: Any coordinate system (automatically normalized)
- **Elevation data**: Required for all formats
- **Missing data**: Automatically handled (NaN, nodata values removed)

## ⚙️ Advanced Configuration

### Performance Optimization

**For Large Files (>1GB)**:
```python
# In sidebar settings:
Quality vs Speed: 70-80 (faster processing)
Max Points: 300,000-500,000
Smart Processing: Enabled
```

**For High Detail**:
```python
# In sidebar settings:
Quality vs Speed: 20-40 (higher quality)
Max Points: 800,000-1,200,000
Advanced ML Features: Enabled
```

**For Memory-Limited Systems**:
```python
# In sidebar settings:
Max Points: 100,000-300,000
Smart Processing: Enabled
```

### Audio Settings

**Scientific Analysis**:
- Mode: Clean
- Tempo: 60-80 BPM (slower, more detail)
- Duration: 30-45 seconds

**Geographic Exploration**:
- Mode: Spatial  
- Tempo: 100-120 BPM
- Duration: 20-30 seconds

**Research/Publication**:
- Mode: Advanced
- Tempo: 120 BPM
- Duration: 30-60 seconds

## 🎧 Audio Interpretation Guide

### What You're Hearing

**Pitch Mapping**:
- **Higher pitches** = Higher elevations (mountains, ridges)
- **Lower pitches** = Lower elevations (valleys, plains)

**Spatial Audio**:
- **Left audio** = Western parts of terrain
- **Right audio** = Eastern parts of terrain

**Special Sounds**:
- **Sharp beeps/chimes** = Anomalies or unusual features
- **Different instruments** = Different terrain clusters
- **Harmonic chords** = Flat surfaces
- **Quick pings** = Sharp edges or ridges

### Listening Tips
- **Use headphones** for best spatial experience
- **Close your eyes** and imagine flying over the terrain
- **Listen for patterns** that repeat - these might be geological formations
- **Focus on pitch changes** rather than individual notes

## 🛠️ Module Architecture

### Core Components

```
sonific/
├── app.py                    # Main Streamlit application
├── data_processor.py         # ML processing & feature extraction
├── loader_factory.py         # Multi-format file loading
├── audio_generator.py        # Audio synthesis orchestration
├── instrument_synthesizer.py # Sound synthesis engines
├── hrtf_processor.py         # 3D spatial audio positioning
├── multitrack_renderer.py    # Professional audio mixing
├── semantic_labeler.py       # Terrain classification
└── ai_agent.py              # LLM-powered analysis assistant
```

### Data Flow
1. **File Loading** → `loader_factory.py`
2. **3D Processing** → `data_processor.py` 
3. **ML Analysis** → Clustering, anomaly detection, surface classification
4. **Audio Synthesis** → `audio_generator.py` coordinates all components
5. **Spatial Rendering** → `hrtf_processor.py` + `multitrack_renderer.py`
6. **User Interface** → `app.py` Streamlit frontend
7. **AI Analysis** → `ai_agent.py` provides expert feedback

## 🔧 Troubleshooting

### Common Issues

**"Processing failed: all input arrays must have the same shape"**
- **Cause**: Malformed data file with inconsistent structure
- **Solution**: The system automatically handles this with data cleaning
- **Prevention**: Check data file integrity before upload

**"Audio generation failed"**
- **Cause**: Usually parameter type conversion issues
- **Solution**: Restart the application, try different settings
- **Prevention**: Use recommended file formats and sizes

**"LLM connection failed"**
- **Cause**: AI assistant can't connect to  LLM server
- **Solution**: Either check the connection or use offline mode (works perfectly)
- **Note**: Offline mode provides expert responses without LLM

**Memory errors with large files**
- **Cause**: Insufficient RAM for large datasets
- **Solution**: Reduce "Max Points" setting or enable "Smart Processing"
- **Tip**: 16GB RAM recommended for files >1GB

### Performance Tips

**Slow processing**:
- Increase "Quality vs Speed" slider
- Reduce "Max Points" 
- Enable "Smart Processing"

**Audio quality issues**:
- Use headphones for spatial audio
- Try different sonification modes
- Adjust tempo and duration settings

**File format issues**:
- Ensure coordinate data is complete (X, Y, Z)
- Check for missing or NaN values
- Verify file isn't corrupted

## 📝 Example Workflows

### Workflow 1: Scientific Terrain Analysis
1. Upload DEM TIFF file
2. Enable "Advanced ML Features"
3. Set high detail processing (low quality slider)
4. Use "Clean" sonification mode
5. Generate 45-second audio at 60 BPM
6. Analyze with AI assistant for scientific insights

### Workflow 2: Accessibility Application
1. Upload any 3D format
2. Use "Clean" mode for continuous audio
3. Set moderate tempo (80-100 BPM)
4. Focus on pitch changes for navigation
5. Use headphones for spatial positioning

### Workflow 3: Research Publication
1. Upload high-quality dataset
2. Enable all advanced features
3. Use "Advanced" multi-track mode
4. Generate multiple durations for comparison
5. Export both WAV and MIDI
6. Get AI analysis for publication description

## 🏆 Advanced Features

### Machine Learning Components
- **DBSCAN Clustering**: Density-based terrain segmentation
- **Isolation Forest**: Anomaly detection in 3D space
- **Surface Classification**: Flat, curved, edge detection
- **Geometric Feature Extraction**: Planes, edges, corners
- **Normal Vector Computation**: Surface orientation analysis

### Audio Processing
- **HRTF Spatial Audio**: Head-related transfer functions for 3D positioning
- **Multi-instrument Synthesis**: Piano, violin, flute, bells, drums, pads
- **Professional Mixing**: 4-track system with volume balancing
- **Real-time Processing**: Efficient algorithms for large datasets

### AI Integration
- **LLM Analysis**: GPT-compatible models for audio interpretation
- **Expert Fallbacks**: Pre-built responses for offline use
- **Audio Feature Extraction**: Spectral analysis, tempo detection
- **Contextual Guidance**: Dataset-specific recommendations




## 🔮 Future Development

- **Real-time streaming**: Live data sonification
- **Additional formats**: Support for more 3D data types



