import streamlit as st
import numpy as np
import tempfile
import os
import pandas as pd
import time 
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from loader_factory import LoaderFactory  
from audio_generator import AudioGenerator
from ai_agent import SonificAIAgent

st.set_page_config(
    page_title="Sonific", 
    layout="wide",
    page_icon="🎵",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        color: #333333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: white;
        color: #333333;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: white;
        color: #333333;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background: white;
        color: #333333;
        border: 1px solid #17a2b8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

class Sonific:
    
    def __init__(self):
        self.dp = DataProcessor()
        self.lf = LoaderFactory() 
        self.ag = AudioGenerator()
        self.ai_agent = None
    
    def display_header(self):
        st.markdown("""
        <div class="main-header">
            <h1>Sonific</h1>
            <h3>ML-Enhanced 3D Data Sonification Platform</h3>
            <p>Transform your 3D datasets into intelligent audio experiences through machine learning and spatial audio</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_sidebar(self):
        with st.sidebar:
            st.markdown("### Data Upload")
            
            uploaded_file = st.file_uploader(
                "Choose your 3D dataset",
                type=['tif', 'tiff', 'las', 'ply', 'csv', 'txt'],
                help="Supports: DEM (TIFF), LiDAR (LAS), Point Clouds (PLY), CSV with X,Y,Z columns"
            )
            
            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 100:
                    st.warning(f"Large file: {file_size_mb:.1f}MB")
                else:
                    st.success(f"File loaded: {file_size_mb:.1f}MB")
            
            st.markdown("---")
            st.markdown("### Processing Settings")
            
            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 500:
                    default_downsample = 80
                    default_max_points = 300000
                elif file_size_mb > 100:
                    default_downsample = 60
                    default_max_points = 500000
                else:
                    default_downsample = 40
                    default_max_points = 800000
            else:
                default_downsample = 50
                default_max_points = 500000
            
            downsample = st.slider(
                "Quality vs Speed", 
                10, 100, default_downsample,
                help="Lower = Higher Quality, Higher = Faster Processing"
            )
            
            max_points = st.selectbox(
                "Max Points to Process",
                [100000, 300000, 500000, 800000, 1200000, 2000000],
                index=2,
                help="More points = better detail but slower processing"
            )
            
            adaptive_skip = st.checkbox(
                "Smart Processing", 
                value=True, 
                help="Automatically optimize settings for your file size"
            )
            
            enable_advanced = st.checkbox(
                "Advanced ML Features",
                value=True,
                help="Enable surface analysis, geometric features, and enhanced audio mapping"
            )
            
            st.markdown("---")
            st.markdown("### Audio Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                tempo = st.slider("Tempo", 60, 180, 120, help="BPM - faster = more compressed timeline")
            with col2:
                duration = st.slider("Duration", 10, 60, 25, help="Seconds of audio output")
            
            scan_mode = st.selectbox(
                "Scanning Strategy", 
                ["elevation", "spatial_sweep", "advanced"],
                format_func=lambda x: "Elevation-based" if x == "elevation" else "Spatial Sweep" if x == "spatial_sweep" else "Advanced Multi-track",
                help="How to traverse through your 3D data"
            )
            
            st.markdown("---")
            
            with st.expander("Need Help?"):
                st.markdown("""
                **File Formats:**
                - **TIFF/TIF**: Digital Elevation Models
                - **LAS**: LiDAR point clouds  
                - **PLY**: 3D point clouds
                - **CSV/TXT**: X,Y,Z coordinate data
                
                **Performance Tips:**
                - Large files (>1GB): Increase quality slider
                - Want detail: Decrease quality slider, increase max points
                - Memory issues: Reduce max points
                """)
        
        return uploaded_file, tempo, duration, downsample, scan_mode, max_points, adaptive_skip, enable_advanced
    
    def process_uploaded_file(self, uploaded_file, downsample, max_points, adaptive_skip, enable_advanced):
        if uploaded_file is None:
            return None
            
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text(f"Loading {file_size_mb:.1f}MB file...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            chunk_size = 8192
            bytes_written = 0
            
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
                bytes_written += len(chunk)
                progress = min(bytes_written / uploaded_file.size, 1.0)
                progress_bar.progress(progress * 0.2)
            
            temp_path = tmp_file.name
        
        try:
            progress_text.text("Analyzing 3D structure...")
            progress_bar.progress(0.3)
            
            if hasattr(self.dp, 'process_large_dataset') and file_size_mb > 100:
                if enable_advanced:
                    result = self.dp.process_large_dataset(temp_path, skip=downsample, max_points=max_points)
                else:
                    result = self.dp.process_large_dataset(temp_path, skip=downsample, max_points=max_points)
            else:
                if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
                    if enable_advanced:
                        result = self.dp.process_dataset(temp_path, skip=downsample)
                    else:
                        result = self.dp.process_dataset(temp_path, skip=downsample)
                else:
                    points = self.lf.load_any_format(temp_path, skip=downsample)
                    
                    if len(points) > max_points:
                        indices = np.random.choice(len(points), max_points, replace=False)
                        points = points[indices]
                    
                    progress_text.text("Running ML clustering algorithms...")
                    progress_bar.progress(0.5)
                    
                    clusters = self.dp.perform_clustering(points)
                    
                    progress_text.text("Computing geometric features...")
                    progress_bar.progress(0.7)
                    
                    normals = self.dp.compute_normals(points)
                    anomalies = self.dp.detect_anomalies(points)
                    
                    if enable_advanced:
                        surface_types = self.dp.classify_surface_types(points, normals)
                        geometric_features = self.dp.detect_geometric_features(points, normals)
                        
                        result = {
                            'points': points,
                            'clusters': clusters,
                            'normals': normals,
                            'surface_types': surface_types,
                            'geometric_features': geometric_features,
                            'anomalies': anomalies,
                            'stats': {
                                'n_points': len(points),
                                'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                                'n_anomalies': anomalies.sum(),
                                'elevation_range': np.ptp(points[:,2]),
                                'n_planes': len(geometric_features['planes']),
                                'n_edges': len(geometric_features['edges']),
                                'n_corners': len(geometric_features['corners']),
                                'file_size_mb': file_size_mb,
                                'advanced_features': True
                            }
                        }
                    else:
                        result = {
                            'points': points,
                            'clusters': clusters,
                            'normals': normals,
                            'anomalies': anomalies,
                            'stats': {
                                'n_points': len(points),
                                'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                                'n_anomalies': anomalies.sum(),
                                'elevation_range': np.ptp(points[:,2]),
                                'file_size_mb': file_size_mb,
                                'advanced_features': False
                            }
                        }
            
            progress_bar.progress(1.0)
            progress_text.text("Processing complete!")
            
            time.sleep(0.5)
            progress_text.empty()
            progress_bar.empty()
            
            os.unlink(temp_path)
            return result
            
        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def create_3d_visualization(self, result):
        points = result['points']
        clusters = result['clusters']
        
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            vis_points = points[indices]
            vis_clusters = clusters[indices]
        else:
            vis_points = points
            vis_clusters = clusters
        
        fig = go.Figure(data=[go.Scatter3d(
            x=vis_points[:, 0],
            y=vis_points[:, 1], 
            z=vis_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=vis_clusters,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster ID")
            ),
            text=[f'Cluster: {c}' for c in vis_clusters],
            hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Z:</b> %{z}<br>%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title="3D Data Visualization (Sample)",
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate", 
                zaxis_title="Elevation",
                bgcolor="rgba(0,0,0,0)"
            ),
            height=500,
            margin=dict(r=0, b=0, l=0, t=40)
        )
        
        return fig
    
    def display_data_analysis(self, result):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Dataset Analysis")
            
            stats = result['stats']
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{stats['n_points']:,} Total Points</h4>
                <p>Spatial data points processed</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{stats['n_clusters']} Terrain Features</h4>
                <p>Distinct structures detected by ML</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{stats['n_anomalies']} Anomalies</h4>
                <p>Unusual or outlier data points</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{stats['elevation_range']:.1f}m Range</h4>
                <p>Total elevation variation</p>
            </div>
            """, unsafe_allow_html=True)
            
            if stats.get('advanced_features', False):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{stats.get('n_planes', 0)} Flat Surfaces</h4>
                    <p>Detected planar features</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{stats.get('n_edges', 0)} Sharp Edges</h4>
                    <p>Detected edge features</p>
                </div>
                """, unsafe_allow_html=True)
            
            if 'downsampled' in stats and stats['downsampled']:
                st.markdown("""
                <div class="info-box">
                    <strong>Note:</strong> Data was intelligently downsampled for optimal performance.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 3D Visualization")
            
            try:
                fig = self.create_3d_visualization(result)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("3D visualization unavailable for this dataset size")
                
                elevations = result['points'][:, 2]
                fig_hist = px.histogram(
                    x=elevations, 
                    nbins=50,
                    title="Elevation Distribution",
                    labels={'x': 'Elevation (m)', 'y': 'Count'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    def display_cluster_analysis(self, result):
        st.markdown("### Feature Analysis")
        
        clusters = result['clusters']
        points = result['points']
        anomalies = result['anomalies']
        
        col1, col2 = st.columns(2)
        
        with col1:
            unique_clusters = np.unique(clusters[clusters != -1])
            if len(unique_clusters) > 0:
                cluster_counts = [np.sum(clusters == c) for c in unique_clusters]
                cluster_df = pd.DataFrame({
                    'Cluster_ID': unique_clusters,
                    'Point_Count': cluster_counts,
                    'Percentage': np.array(cluster_counts) / len(points) * 100
                })
                
                if len(cluster_df) > 10:
                    cluster_df = cluster_df.nlargest(10, 'Point_Count')
                    st.caption("Top 10 clusters by size")
                
                fig_clusters = px.bar(
                    cluster_df, 
                    x='Cluster_ID', 
                    y='Point_Count',
                    title="Detected Terrain Features",
                    labels={'Point_Count': 'Points', 'Cluster_ID': 'Feature ID'}
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
            else:
                st.warning("No distinct clusters detected in this dataset")
        
        with col2:
            if len(unique_clusters) > 0:
                elevation_stats = []
                for cluster_id in unique_clusters[:10]:
                    cluster_points = points[clusters == cluster_id]
                    if len(cluster_points) > 0:
                        elevation_stats.append({
                            'Feature_ID': f"Feature_{cluster_id}",
                            'Avg_Elevation': np.mean(cluster_points[:, 2]),
                            'Elevation_Range': np.ptp(cluster_points[:, 2]),
                            'Points': len(cluster_points)
                        })
                
                if elevation_stats:
                    stats_df = pd.DataFrame(elevation_stats)
                    fig_elev = px.scatter(
                        stats_df,
                        x='Avg_Elevation',
                        y='Elevation_Range', 
                        size='Points',
                        title="Feature Characteristics",
                        labels={'Avg_Elevation': 'Mean Elevation (m)', 'Elevation_Range': 'Elevation Spread (m)'},
                        hover_data=['Feature_ID']
                    )
                    st.plotly_chart(fig_elev, use_container_width=True)
                    
                    with st.expander("Detailed Feature Statistics"):
                        st.dataframe(stats_df, use_container_width=True)
        
        if result['stats'].get('advanced_features', False):
            self.display_advanced_features(result)
    
    def display_advanced_features(self, result):
        st.markdown("### Advanced Feature Analysis")
        
        surface_types = result.get('surface_types', [])
        geometric_features = result.get('geometric_features', {})
        
        if surface_types:
            surface_counts = pd.Series(surface_types).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_surface = px.pie(
                    values=surface_counts.values,
                    names=surface_counts.index,
                    title="Surface Type Distribution"
                )
                st.plotly_chart(fig_surface, use_container_width=True)
            
            with col2:
                if geometric_features:
                    feature_summary = {
                        'Feature Type': ['Planes', 'Edges', 'Corners'],
                        'Count': [
                            len(geometric_features.get('planes', [])),
                            len(geometric_features.get('edges', [])),
                            len(geometric_features.get('corners', []))
                        ]
                    }
                    
                    fig_features = px.bar(
                        pd.DataFrame(feature_summary),
                        x='Feature Type',
                        y='Count',
                        title="Geometric Features Detected"
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
    
    def generate_sonification(self, result, tempo, duration, scan_mode):
        points = result['points']
        clusters = result['clusters']
        anomalies = result['anomalies']
        
        if scan_mode == "advanced" and result['stats'].get('advanced_features', False):
            sonification_result = self.ag.generate_full_featured_sonification(result, tempo, duration)
        elif scan_mode == "elevation":
            sonification_result = self.ag.generate_clean_sonification(points, clusters, anomalies, tempo, duration)
        else:
            sonification_result = self.ag.generate_spatial_sonification(points, clusters, anomalies, tempo, duration)
        
        return sonification_result
    
    def display_audio_section(self, result, tempo, duration, scan_mode):
        st.markdown("### Generate Sonification")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("GENERATE AUDIO FILES", type="primary", use_container_width=True):
                with st.spinner(f"Creating {scan_mode} sonification..."):
                    try:
                        audio_start = time.time()
                        sonification_result = self.generate_sonification(result, tempo, duration, scan_mode)
                        audio_time = time.time() - audio_start
                        
                        st.session_state['sonification_result'] = sonification_result
                        st.session_state['audio_time'] = audio_time
                        st.session_state['audio_settings'] = {
                            'mode': scan_mode, 'tempo': tempo, 'duration': duration
                        }
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Audio generation failed: {str(e)}")
                        st.info("Try reducing the file size or adjusting processing settings")
        
        with col2:
            st.markdown("**Current Settings:**")
            st.write(f"**Mode**: {scan_mode.title()}")
            st.write(f"**Tempo**: {tempo} BPM")
            st.write(f"**Duration**: {duration}s")
            st.write(f"**Points**: {result['stats']['n_points']:,}")
            
            if result['stats'].get('advanced_features', False):
                st.write("**Advanced Features**: Enabled")
        
        if 'sonification_result' in st.session_state:
            self.display_audio_results()
    
    def display_audio_results(self):
        sonification_result = st.session_state['sonification_result']
        
        wav_file = sonification_result['wav_file']
        midi_file = sonification_result['midi_file']
        
        if os.path.exists(wav_file) and os.path.exists(midi_file):
            
            st.markdown("""
            <div class="success-box">
                <h4>Audio Generation Complete!</h4>
                <p>Your 3D data has been transformed into audio. Listen below or download the files.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Listen Now")
            with open(wav_file, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")
            
            st.markdown("#### Download Files")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                with open(wav_file, "rb") as f:
                    st.download_button(
                        "Download WAV",
                        f.read(),
                        file_name=os.path.basename(wav_file),
                        mime="audio/wav",
                        use_container_width=True,
                        help="High-quality audio file for listening"
                    )
            
            with col2:
                with open(midi_file, "rb") as f:
                    st.download_button(
                        "Download MIDI", 
                        f.read(),
                        file_name=os.path.basename(midi_file),
                        mime="audio/midi",
                        use_container_width=True,
                        help="Musical notation file for editing"
                    )
            
            with col3:
                if st.button("Generate New", use_container_width=True):
                    if 'sonification_result' in st.session_state:
                        del st.session_state['sonification_result']
                    st.rerun()
            
            if 'audio_time' in st.session_state:
                st.info(f"Generated in {st.session_state['audio_time']:.1f} seconds")
    
    def display_ai_agent_interface(self):
        st.markdown("### AI Audio Analysis Assistant")
        
        if self.ai_agent is None:
            self.ai_agent = SonificAIAgent(self)
        
        # Display connection status
        if hasattr(self.ai_agent, 'llm_available'):
            if self.ai_agent.llm_available:
                st.success("🤖 AI Assistant: Connected to LM Studio")
            else:
                st.warning("🤖 AI Assistant: Running in offline mode (Expert fallback responses)")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (role, message) in enumerate(st.session_state.chat_history):
            with st.chat_message(role):
                st.write(message)
        
        # Chat input
        if user_query := st.chat_input("Ask about your sonification quality and audio analysis..."):
            st.session_state.chat_history.append(("user", user_query))
            
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your audio and data..."):
                    # Get current analysis results and audio file
                    analysis_results = st.session_state.get('processed_result', {})
                    audio_file_path = None
                    
                    # Get audio file path if sonification exists
                    if 'sonification_result' in st.session_state:
                        sonification_result = st.session_state['sonification_result']
                        audio_file_path = sonification_result.get('wav_file')
                    
                    try:
                        # Use the new audio analysis method
                        response = self.ai_agent.process_user_query_with_audio_analysis(
                            user_query, analysis_results, audio_file_path
                        )
                        st.write(response)
                        st.session_state.chat_history.append(("assistant", response))
                        
                    except Exception as e:
                        error_msg = f"Analysis error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(("assistant", error_msg))
        
        # Quick action buttons focused on audio analysis
        st.markdown("#### Quick Audio Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎧 Analyze Audio Quality"):
                example_query = "Please provide your expert opinion on the quality of my generated sonification. How well does it represent the terrain data?"
                st.session_state.chat_history.append(("user", example_query))
                st.rerun()
        
        with col2:
            if st.button("🔍 What Am I Hearing?"):
                example_query = "Analyze my audio file and explain what specific sounds I should be listening for and what they mean in terms of my terrain data."
                st.session_state.chat_history.append(("user", example_query))
                st.rerun()
        
        with col3:
            if st.button("🎛️ Improve Settings"):
                example_query = "Based on your analysis of my audio file and dataset, what specific parameter adjustments would improve the sonification quality?"
                st.session_state.chat_history.append(("user", example_query))
                st.rerun()

    
    def display_interpretation_guide(self):
        st.markdown("### Audio Interpretation Guide")
        
        with st.expander("How to understand your sonification", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Musical Elements:**
                - **Higher pitch** → Higher elevation
                - **Lower pitch** → Lower elevation/valleys  
                - **Loud/bright notes** → Anomalous features
                - **Chord progressions** → Grouped terrain features
                - **Left audio** → Western part of area
                - **Right audio** → Eastern part of area
                """)
            
            with col2:
                st.markdown("""
                **Terrain Features:**
                - **Rising melodies** → Hills, ridges, slopes
                - **Falling melodies** → Valleys, depressions
                - **Sudden pitch changes** → Cliffs, sharp edges
                - **Repeated patterns** → Regular geological structures
                - **Percussion sounds** → Detected anomalies
                - **Different instruments** → Distinct terrain types
                """)
            
            st.markdown("""
            **Listening Tips:**
            - Use **headphones** for best spatial audio experience
            - **Close your eyes** and imagine flying over the terrain
            - Listen for **patterns** that repeat - these might be geological formations
            - **Sharp sounds** often indicate interesting features worth investigating
            """)
    
    def display_welcome_screen(self):
        st.markdown("### Welcome to Sonific")
        
        st.markdown("""
        <div class="info-box">
            <h4>Transform 3D Data into Sound</h4>
            <p>Sonific uses machine learning to convert complex 3D datasets into meaningful audio experiences. Perfect for research, accessibility, and data exploration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **ML-Powered Analysis**
            - Automatic feature detection
            - Intelligent clustering
            - Anomaly identification
            """)
        
        with col2:
            st.markdown("""
            **Smart Audio Mapping**
            - Spatial 3D audio positioning
            - Musical scale harmonies
            - Multiple output formats
            """)
        
        with col3:
            st.markdown("""
            **Multiple Formats**
            - DEM TIFF files (up to 10GB)
            - LiDAR LAS point clouds
            - CSV coordinate data
            """)
        
        st.markdown("### Supported File Formats")
        
        format_data = {
            "Format": ["DEM TIFF", "LiDAR LAS", "Point Cloud PLY", "CSV/TXT"],
            "Description": [
                "Digital elevation models (.tif, .tiff)",
                "Laser scan point clouds (.las)",  
                "3D point cloud data (.ply)",
                "X,Y,Z coordinate tables (.csv, .txt)"
            ],
            "Max Size": ["10GB+", "2GB", "1GB", "500MB"],
            "Use Case": ["Terrain mapping", "Building scanning", "3D models", "Custom datasets"]
        }
        
        st.table(pd.DataFrame(format_data))
        
        with st.expander("Performance Tips"):
            st.markdown("""
            **For Best Results:**
            - **Large files (>1GB)**: Enable smart processing and use higher quality setting
            - **High detail needed**: Increase max points and lower quality setting  
            - **Fast preview**: Use higher quality setting with fewer max points
            - **Memory limitations**: Keep max points under 500K
            - **Lunar/planetary data**: Our system is optimized for elevation models like LRO data
            """)
    
    def run(self):
        self.display_header()
        
        uploaded_file, tempo, duration, downsample, scan_mode, max_points, adaptive_skip, enable_advanced = self.display_sidebar()
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > 1000:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Large File Alert:</strong> {file_size_mb:.1f}MB file detected. 
                    Processing may take 2-5 minutes. Consider enabling Smart Processing.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <strong>File Ready:</strong> {uploaded_file.name} ({file_size_mb:.1f}MB)
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Process Your Dataset")
            
            if st.button("ANALYZE 3D DATA", type="primary", use_container_width=True):
                
                with st.spinner("System is processing your 3D data..."):
                    try:
                        process_start = time.time()
                        result = self.process_uploaded_file(uploaded_file, downsample, max_points, adaptive_skip, enable_advanced)
                        process_time = time.time() - process_start
                        
                        st.session_state['processed_result'] = result
                        st.session_state['process_time'] = process_time
                        
                        st.success(f"Analysis complete in {process_time:.1f} seconds!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
                        st.info("Try adjusting the processing settings in the sidebar")
            
            if 'processed_result' in st.session_state:
                result = st.session_state['processed_result']
                
                if 'process_time' in st.session_state:
                    st.info(f"Processing completed in {st.session_state['process_time']:.1f} seconds")
                
                st.markdown("---")
                self.display_data_analysis(result)
                
                st.markdown("---")
                self.display_cluster_analysis(result)
                
                st.markdown("---")
                self.display_audio_section(result, tempo, duration, scan_mode)
                
                st.markdown("---")
                self.display_ai_agent_interface()
                
                st.markdown("---")
                self.display_interpretation_guide()
                
        else:
            self.display_welcome_screen()

if __name__ == "__main__":
    app = Sonific()
    app.run()
