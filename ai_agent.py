# ai_agent.py
import os
import librosa
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory  # ← CORRECT IMPORT
from langchain_core.runnables.history import RunnableWithMessageHistory

class SonificAIAgent:
    
    def __init__(self, sonification_system=None):
        self.system = sonification_system
        self.setup_llm()
        self.setup_memory()
        self.setup_system_prompt()
        
    def setup_llm(self):
        """Initialize LM Studio connection with robust error handling"""
        try:
            self.llm = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                model="local-model",
                temperature=0.7,
                max_tokens=2000,  # Increased for longer responses
                timeout=60,       # Increased timeout
                request_timeout=60,  # Additional timeout parameter
                streaming=False   # Disable streaming for more reliable responses
            )
            
            # Test connection with a simple query
            test_response = self.llm.invoke([HumanMessage(content="Hello, respond with 'OK' if you're working.")])
            if test_response and test_response.content:
                print("✅ LM Studio connection successful")
                self.llm_available = True
            else:
                raise Exception("No response from LM Studio")
                
        except Exception as e:
            print(f"⚠️ LM Studio connection failed: {str(e)}")
            print("AI Agent will run in offline mode with pre-built responses")
            self.llm = None
            self.llm_available = False

    
    def setup_memory(self):
        """Initialize modern chat message history"""
        self.store = {}
        
        def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]
        
        self.get_session_history = get_session_history
        
        if self.llm:
            self.chain_with_history = RunnableWithMessageHistory(
                self.llm,
                get_session_history,
                input_messages_key="messages",
                history_messages_key="history",
            )
    
    def setup_system_prompt(self):
        """Define the AI agent's specialized audio analysis persona"""
        self.system_message = SystemMessage(content="""You are an expert Sonific3D Audio Analysis Assistant specialized in evaluating and interpreting 3D data sonifications.

Your primary role is to:
1. **Analyze generated audio files** and provide expert opinions on sonification quality
2. **Interpret audio features** in the context of the underlying 3D dataset  
3. **Help users understand** what they're hearing and how it relates to their data
4. **Suggest improvements** for better data representation through sound

**Audio Analysis Expertise:**
- Spectral analysis (frequency distribution, spectral centroid, bandwidth)
- Temporal patterns (rhythm, pacing, duration optimization)
- Spatial audio effectiveness (stereo separation, HRTF positioning quality)
- Dynamic range and amplitude characteristics
- Instrument separation and timbral clarity

**Sonification Mapping Knowledge:**
- Pitch → Elevation mapping effectiveness
- Spatial positioning → Geographic accuracy
- Instrument timbres → Cluster differentiation quality
- Anomaly alert sounds → Detection clarity
- Multi-track mixing → Information layering success

**Always provide:**
- Specific observations about audio characteristics
- How audio features relate to data patterns
- Concrete suggestions for improvement
- Educational explanations of sonification principles""")

    def analyze_audio_file(self, audio_file_path):
        """Comprehensive audio analysis for sonification evaluation"""
        if not os.path.exists(audio_file_path):
            return {"error": "Audio file not found"}
            
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_file_path, sr=44100)
            
            if len(audio) == 0:
                return {"error": "Empty audio file"}
            
            # Extract comprehensive audio features
            features = {
                'basic_stats': {
                    'duration': float(len(audio) / sr),
                    'sample_rate': int(sr),
                    'total_samples': len(audio),
                    'rms_energy': float(np.sqrt(np.mean(audio**2))),
                    'dynamic_range': float(np.max(audio) - np.min(audio))
                },
                'spectral_features': self._extract_spectral_features(audio, sr),
                'temporal_features': self._extract_temporal_features(audio, sr),
                'spatial_features': self._analyze_stereo_characteristics(audio),
                'sonification_quality': self._assess_sonification_quality(audio, sr)
            }
            
            return features
            
        except Exception as e:
            return {"error": f"Audio analysis failed: {str(e)}"}
    
    def _extract_spectral_features(self, audio, sr):
        """Extract frequency-domain features relevant to sonification"""
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Pitch analysis for elevation mapping assessment
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=80, fmax=2000)
            valid_f0 = f0[voiced_flag]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'pitch_range': {
                    'min_freq': float(np.min(valid_f0)) if len(valid_f0) > 0 else 0,
                    'max_freq': float(np.max(valid_f0)) if len(valid_f0) > 0 else 0,
                    'pitch_variation': float(np.std(valid_f0)) if len(valid_f0) > 0 else 0
                },
                'voiced_percentage': float(np.sum(voiced_flag) / len(voiced_flag) * 100)
            }
        except:
            return {'error': 'Spectral analysis failed'}
    
    def _extract_temporal_features(self, audio, sr):
        """Analyze temporal characteristics for sonification pacing"""
        try:
            # Tempo and beat analysis
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Zero crossing rate (indicates texture changes)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # RMS energy over time (amplitude envelope)
            rms = librosa.feature.rms(y=audio)[0]
            
            return {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'avg_beat_interval': float(len(audio) / sr / len(beats)) if len(beats) > 0 else 0,
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'rms_mean': float(np.mean(rms)),
                'rms_variation': float(np.std(rms)),
                'dynamic_contrast': float(np.max(rms) - np.min(rms))
            }
        except:
            return {'error': 'Temporal analysis failed'}
    
    def _analyze_stereo_characteristics(self, audio):
        """Assess spatial audio quality for 3D positioning"""
        try:
            if audio.ndim == 1:
                return {'stereo_separation': 0, 'note': 'Mono audio - no spatial positioning'}
            
            # Assuming stereo input, convert to stereo if mono
            if audio.ndim == 2:
                left_channel = audio[:, 0]
                right_channel = audio[:, 1]
                
                # Calculate stereo separation metrics
                correlation = np.corrcoef(left_channel, right_channel)[0, 1]
                left_energy = np.mean(left_channel**2)
                right_energy = np.mean(right_channel**2)
                
                return {
                    'stereo_separation': float(1 - abs(correlation)),
                    'left_right_balance': float(left_energy / (left_energy + right_energy)),
                    'channel_correlation': float(correlation),
                    'spatial_width': float(abs(left_energy - right_energy))
                }
            
            return {'stereo_separation': 0, 'note': 'Could not analyze stereo characteristics'}
            
        except:
            return {'error': 'Stereo analysis failed'}
    
    def _assess_sonification_quality(self, audio, sr):
        """Evaluate specific sonification effectiveness metrics"""
        try:
            # Assess frequency distribution for elevation mapping
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            freq_bins = librosa.fft_frequencies(sr=sr)
            
            # Calculate spectral statistics
            spectral_flux = np.mean(np.diff(magnitude, axis=1)**2)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
            
            # Assess timbral diversity (important for cluster differentiation)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_diversity = np.mean(np.std(mfccs, axis=1))
            
            return {
                'spectral_flux': float(spectral_flux),
                'spectral_flatness': float(spectral_flatness),
                'timbral_diversity': float(mfcc_diversity),
                'frequency_coverage': {
                    'low_freq_energy': float(np.sum(magnitude[:len(freq_bins)//4])),
                    'mid_freq_energy': float(np.sum(magnitude[len(freq_bins)//4:3*len(freq_bins)//4])),
                    'high_freq_energy': float(np.sum(magnitude[3*len(freq_bins)//4:]))
                }
            }
        except:
            return {'error': 'Quality assessment failed'}

    # In ai_agent.py, update this method:
    def process_user_query_with_audio_analysis(self, query, analysis_results=None, audio_file_path=None):
        """Main method: Analyze audio and provide expert opinion"""
        
        # Build comprehensive context (keep existing context building code)
        context_parts = []
        
        # Dataset context
        if analysis_results:
            stats = analysis_results.get('stats', {})
            context_parts.append(f"""**Dataset Information:**
    - Total points: {stats.get('n_points', 0):,}
    - Terrain clusters detected: {stats.get('n_clusters', 0)}
    - Elevation range: {stats.get('elevation_range', 0):.1f}m
    - Anomalies found: {stats.get('n_anomalies', 0)}
    - Advanced features: {'Enabled' if stats.get('advanced_features', False) else 'Disabled'}""")
        
        # Audio analysis context
        audio_features = None
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"🎧 Analyzing audio file: {audio_file_path}")
            audio_features = self.analyze_audio_file(audio_file_path)
            
            if "error" not in audio_features:
                context_parts.append(f"""**Audio Analysis Results:**
    - Duration: {audio_features['basic_stats']['duration']:.1f}s
    - Pitch range: {audio_features['spectral_features']['pitch_range']['min_freq']:.0f} - {audio_features['spectral_features']['pitch_range']['max_freq']:.0f} Hz
    - Spectral centroid: {audio_features['spectral_features']['spectral_centroid_mean']:.0f} Hz
    - Tempo: {audio_features['temporal_features']['tempo']:.1f} BPM
    - Stereo separation: {audio_features['spatial_features']['stereo_separation']:.2f}
    - Timbral diversity: {audio_features['sonification_quality']['timbral_diversity']:.3f}""")
            else:
                print(f"❌ Audio analysis failed: {audio_features['error']}")
        else:
            print("⚠️ No audio file found for analysis")
        
        context_info = "\n\n".join(context_parts)
        
        # Try AI response with better error handling and timeout
        if self.llm_available:
            try:
                print("🤖 Sending query to LM Studio...")
                
                # Create the message
                full_prompt = f"""{context_info}

    **User Query:** {query}

    Please provide your expert analysis of this sonification, focusing on:
    1. How well the audio features represent the underlying data
    2. Quality assessment of the sonification techniques used  
    3. Specific observations about what the user would hear
    4. Recommendations for improvement or better interpretation"""

                # Use invoke instead of more complex chains
                messages = [self.system_message, HumanMessage(content=full_prompt)]
                
                print("🔄 Waiting for LM Studio response...")
                response = self.llm.invoke(messages, config={"timeout": 60})
                
                print("✅ Received response from LM Studio")
                
                # Add automated insights
                insights = self._generate_audio_insights(audio_features, analysis_results)
                final_response = response.content
                
                if insights:
                    final_response += f"\n\n**Additional Technical Insights:**\n"
                    for insight in insights:
                        final_response += f"• {insight}\n"
                
                return final_response
                
            except Exception as e:
                print(f"❌ AI model error: {str(e)}")
                print("🔄 Falling back to expert analysis...")
                return self._generate_expert_fallback_response(query, analysis_results, audio_features)
        else:
            print("⚠️ LM Studio not available, using fallback responses")
            return self._generate_expert_fallback_response(query, analysis_results, audio_features)


    def _generate_audio_insights(self, audio_features, analysis_results):
        """Generate automated technical insights about audio quality"""
        if not audio_features or "error" in audio_features:
            return []
            
        insights = []
        
        # Spectral analysis insights
        spectral = audio_features.get('spectral_features', {})
        if spectral.get('spectral_centroid_mean', 0) > 1000:
            insights.append("High spectral centroid indicates bright, high-frequency content - good for representing elevated terrain")
        
        # Spatial audio insights
        spatial = audio_features.get('spatial_features', {})
        if spatial.get('stereo_separation', 0) > 0.3:
            insights.append("Good stereo separation detected - spatial positioning should be clearly audible")
        elif spatial.get('stereo_separation', 0) < 0.1:
            insights.append("Limited stereo separation - consider enhancing spatial audio processing")
        
        # Timbral diversity insights
        quality = audio_features.get('sonification_quality', {})
        if quality.get('timbral_diversity', 0) > 0.5:
            insights.append("High timbral diversity - different terrain clusters should be easily distinguishable")
        elif quality.get('timbral_diversity', 0) < 0.2:
            insights.append("Low timbral diversity - clusters may sound too similar, consider more distinct instruments")
        
        # Temporal insights
        temporal = audio_features.get('temporal_features', {})
        if temporal.get('tempo', 0) > 140:
            insights.append("Fast tempo - may be too rushed for detailed terrain exploration")
        elif temporal.get('tempo', 0) < 80:
            insights.append("Slow tempo - good for detailed analysis but may feel sluggish")
            
        return insights

    def _generate_expert_fallback_response(self, query, analysis_results, audio_features):
        """Expert fallback responses when AI is unavailable"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["opinion", "analysis", "think", "assess", "evaluate"]):
            response = "🎧 **Expert Sonification Analysis:**\n\n"
            
            if audio_features and "error" not in audio_features:
                # Provide detailed audio-based analysis
                duration = audio_features['basic_stats']['duration']
                pitch_range = audio_features['spectral_features']['pitch_range']
                stereo_sep = audio_features['spatial_features']['stereo_separation']
                timbral_div = audio_features['sonification_quality']['timbral_diversity']
                
                response += f"""**Audio Quality Assessment:**
• **Duration ({duration:.1f}s)**: {'Optimal for exploration' if 15 <= duration <= 45 else 'Consider adjusting length'}
• **Pitch Range ({pitch_range['min_freq']:.0f}-{pitch_range['max_freq']:.0f}Hz)**: {'Good elevation mapping' if pitch_range['max_freq'] - pitch_range['min_freq'] > 200 else 'Limited elevation variation'}
• **Spatial Audio**: {'Excellent positioning' if stereo_sep > 0.3 else 'Could improve stereo separation'}
• **Instrument Clarity**: {'Distinct cluster sounds' if timbral_div > 0.4 else 'Clusters may sound similar'}

**What You Should Hear:**
- **Rising pitches** indicate upward terrain (hills, mountains)
- **Left/right panning** shows east/west geographic positioning  
- **Different instruments** represent distinct terrain types found by ML
- **Sharp beeps** highlight anomalies or unusual features

**Recommendations:**
"""
                
                if duration > 45:
                    response += "• Consider shortening duration for better focus\n"
                if stereo_sep < 0.2:
                    response += "• Enable headphones and check HRTF spatial processing\n"
                if timbral_div < 0.3:
                    response += "• Try advanced sonification mode for better instrument separation\n"
                    
            else:
                response += """**General Sonification Principles:**
Your audio should effectively represent:
• **Elevation through pitch** - higher ground = higher tones
• **Geography through stereo** - spatial positioning matches real locations  
• **Terrain types through timbre** - different areas sound like different instruments
• **Anomalies through alerts** - unusual features get attention-grabbing sounds

**Quality Indicators:**
✅ Clear pitch progression following terrain elevation
✅ Distinct left/right positioning for spatial awareness
✅ Different instrument sounds for terrain clusters  
✅ Appropriate pacing for data exploration"""

            return response
            
        elif "improve" in query_lower or "better" in query_lower:
            return """🔧 **Sonification Improvement Recommendations:**

**For Better Clarity:**
• **Increase tempo** if audio feels too slow for exploration
• **Enable advanced mode** for multi-track mixing with better separation
• **Use headphones** to experience full spatial audio positioning
• **Adjust duration** - shorter for overview, longer for detailed exploration

**For Better Terrain Representation:**
• **Higher max points** = more detail but slower processing
• **Advanced ML features** = better terrain classification and sound mapping
• **Spatial sweep mode** = emphasizes geographic relationships

**For Accessibility:**
• Start with **elevation mode** for simplest pitch-to-height mapping
• Use **good stereo headphones** for spatial positioning
• Focus on **pitch changes** rather than individual instruments
• **Close your eyes** and imagine flying over the terrain"""

        return """🎵 **Sonific Audio Analysis Assistant**

I specialize in analyzing your generated sonification files and helping you understand what you're hearing.

**What I can analyze:**
• **Audio quality** - spectral characteristics, dynamic range, spatial separation
• **Sonification effectiveness** - how well sound represents your data
• **Listening guidance** - what specific sounds mean in your terrain
• **Improvement suggestions** - parameter adjustments for better results

**Try asking:**
• "What's your opinion on my sonification quality?"
• "How well does this audio represent my terrain data?"  
• "What should I be listening for in this audio?"
• "How can I improve the clarity of different terrain types?"

**Note:** For full AI analysis, ensure LM Studio is running with a loaded model."""
