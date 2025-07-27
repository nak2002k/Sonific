import numpy as np
import scipy.io.wavfile as wav
from midiutil import MIDIFile
from instrument_synthesizer import InstrumentSynthesizer
from hrtf_processor import HRTFProcessor
from multitrack_renderer import MultiTrackRenderer
from semantic_labeler import SemanticLabeler

class AudioGenerator:
    
    def __init__(self):
        self.scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
        self.scale_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        self.sample_rate = 44100
        
        self.synthesizer = InstrumentSynthesizer(self.sample_rate)
        self.hrtf_processor = HRTFProcessor()
        self.multitrack_renderer = MultiTrackRenderer(self.sample_rate)
        self.semantic_labeler = SemanticLabeler()

    def _ensure_real_audio(self, audio_array):
        """Ensure audio array contains only real numbers"""
        if np.iscomplexobj(audio_array):
            audio_array = np.real(audio_array)
        
        # Replace any NaN or infinite values
        audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return audio_array.astype(np.float32)

    def _safe_normalize(self, data):
        """Safe normalization that handles edge cases"""
        if len(data) == 0:
            return np.array([])
        
        data = np.real(data)  # Ensure real numbers
        data_range = np.ptp(data)
        
        if data_range == 0 or not np.isfinite(data_range):
            return np.zeros_like(data)
        
        min_val = np.min(data)
        if not np.isfinite(min_val):
            return np.zeros_like(data)
        
        normalized = (data - min_val) / data_range
        return np.clip(normalized, 0, 1)

    def normalize_data(self, data):
        """Safe normalization with complex number handling"""
        return self._safe_normalize(data)

    # ============ MODE 1: CLEAN - ALL POINTS AS CONTINUOUS WAVES ============
    def generate_clean_sonification(self, points, clusters, anomalies, tempo=120, duration=25):
        """MODE 1: Process ALL points as overlapping continuous waves"""
        
        try:
            # Process ALL points, no downsampling
            audio_data = self._create_overlapping_continuous_waves(points, clusters, anomalies, duration)
            audio_data = self._ensure_real_audio(audio_data)
            
            # Ensure stereo format
            if audio_data.ndim == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            wav_filename = self.save_wav(audio_data, "clean_sonification.wav")
            midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration, downsample_factor=max(1, len(points)//5000))
            midi_filename = self.save_midi(midi_data, "clean_sonification.mid")
            
            return {
                'wav_file': wav_filename,
                'midi_file': midi_filename,
                'audio_data': audio_data
            }
        except Exception as e:
            raise ValueError(f"Clean sonification failed: {str(e)}")
    
    def _create_overlapping_continuous_waves(self, points, clusters, anomalies, duration):
        """Create continuous waves using ALL points with intelligent overlap"""
        
        left_channel = np.zeros(int(duration * self.sample_rate))
        right_channel = np.zeros(int(duration * self.sample_rate))
        
        # Ensure we have valid points
        if len(points) == 0:
            return np.column_stack((left_channel, right_channel))
        
        # Divide ALL points into time segments
        num_segments = min(len(points), int(duration * 10))  # 10 segments per second max
        points_per_segment = max(1, len(points) // num_segments)
        
        segment_duration = duration / num_segments
        overlap_factor = 0.5  # 50% overlap between segments
        
        # Safe elevation range calculation
        all_elevations = points[:, 2]
        all_elevations = np.real(all_elevations)
        elev_min = np.min(all_elevations[np.isfinite(all_elevations)])
        elev_max = np.max(all_elevations[np.isfinite(all_elevations)])
        elev_range = elev_max - elev_min
        
        if elev_range == 0 or not np.isfinite(elev_range):
            elev_range = 1.0  # Default range
            elev_min = np.mean(all_elevations[np.isfinite(all_elevations)])
        
        for segment_idx in range(num_segments):
            start_point_idx = segment_idx * points_per_segment
            end_point_idx = min(start_point_idx + points_per_segment, len(points))
            
            if start_point_idx >= len(points):
                break
                
            segment_points = points[start_point_idx:end_point_idx]
            segment_anomalies = anomalies[start_point_idx:end_point_idx]
            
            # Safe calculations
            segment_elevations = np.real(segment_points[:, 2])
            valid_elevations = segment_elevations[np.isfinite(segment_elevations)]
            
            if len(valid_elevations) == 0:
                continue
            
            avg_elevation = np.mean(valid_elevations)
            elevation_variation = np.std(valid_elevations)
            
            # Safe frequency calculation
            elev_norm = (avg_elevation - elev_min) / elev_range
            elev_norm = np.clip(elev_norm, 0, 1)
            base_frequency = 200 + elev_norm * 800
            
            # Safe modulation
            var_norm = elevation_variation / elev_range if elev_range > 0 else 0
            freq_modulation = 1 + var_norm * 0.3
            freq_modulation = np.clip(freq_modulation, 0.5, 2.0)
            
            frequency = base_frequency * freq_modulation
            frequency = np.clip(frequency, 80, 2000)  # Audio range
            
            # Generate audio for this segment
            segment_audio_duration = segment_duration * (1 + overlap_factor)
            t = np.linspace(0, segment_audio_duration, int(self.sample_rate * segment_audio_duration), False)
            
            # Create wave with safe operations
            base_wave = np.sin(2 * np.pi * frequency * t)
            harmonic_wave = 0.2 * np.sin(2 * np.pi * frequency * 2 * t)
            
            # Add micro-variations safely
            for i, point in enumerate(segment_points[:min(10, len(segment_points))]):  # Limit iterations
                point_freq = frequency + (i * 5)  # Small variations
                point_influence = 0.05 * np.sin(2 * np.pi * point_freq * t)
                base_wave += point_influence
            
            combined_wave = base_wave + harmonic_wave
            combined_wave = self._ensure_real_audio(combined_wave)
            
            # Anomaly influence
            anomaly_count = np.sum(segment_anomalies)
            if anomaly_count > 0:
                anomaly_factor = 1 + (anomaly_count / len(segment_points)) * 0.5
                anomaly_factor = np.clip(anomaly_factor, 1.0, 2.0)
                combined_wave *= anomaly_factor
            
            # Envelope for smooth transitions
            envelope = np.ones_like(combined_wave)
            fade_samples = int(0.1 * self.sample_rate)
            if len(envelope) > fade_samples * 2:
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            segment_wave = combined_wave * envelope * 0.3
            segment_wave = self._ensure_real_audio(segment_wave)
            
            # Safe spatial positioning
            segment_x_coords = np.real(segment_points[:, 0])
            valid_x = segment_x_coords[np.isfinite(segment_x_coords)]
            
            if len(valid_x) > 0:
                avg_x = np.mean(valid_x)
                x_range = np.ptp(points[:, 0])
                if x_range > 0:
                    x_norm = (avg_x - np.min(points[:, 0])) / x_range
                    x_norm = np.clip(x_norm, 0, 1)
                else:
                    x_norm = 0.5
            else:
                x_norm = 0.5
            
            left_gain = 1 - x_norm
            right_gain = x_norm
            
            # Mix into main channels
            start_time = segment_idx * segment_duration
            start_sample = int(start_time * self.sample_rate)
            end_sample = min(start_sample + len(segment_wave), len(left_channel))
            audio_length = end_sample - start_sample
            
            if audio_length > 0 and start_sample >= 0:
                left_channel[start_sample:end_sample] += segment_wave[:audio_length] * left_gain
                right_channel[start_sample:end_sample] += segment_wave[:audio_length] * right_gain
        
        # Safe normalization
        left_channel = self._ensure_real_audio(left_channel)
        right_channel = self._ensure_real_audio(right_channel)
        
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel /= max_val * 1.2
            right_channel /= max_val * 1.2
        
        return np.column_stack((left_channel, right_channel))

    # ============ MODE 2: SPATIAL - ALL POINTS WITH DENSITY-BASED TIMING ============
    def generate_spatial_sonification(self, points, clusters, anomalies, tempo=100, duration=30):
        """MODE 2: Process ALL points with density-based spatial audio"""
        
        try:
            # Process ALL points with intelligent spatial grouping
            audio_data = self._create_density_based_spatial_audio(points, clusters, anomalies, duration)
            audio_data = self._ensure_real_audio(audio_data)
            
            # Ensure stereo format
            if audio_data.ndim == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            wav_filename = self.save_wav(audio_data, "spatial_sonification.wav")
            midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration, downsample_factor=max(1, len(points)//3000))
            midi_filename = self.save_midi(midi_data, "spatial_sonification.mid")
            
            return {
                'wav_file': wav_filename,
                'midi_file': midi_filename,
                'audio_data': audio_data
            }
        except Exception as e:
            raise ValueError(f"Spatial sonification failed: {str(e)}")
    
    def _create_density_based_spatial_audio(self, points, clusters, anomalies, duration):
        """Use ALL points by creating density-based spatial audio"""
        
        left_channel = np.zeros(int(duration * self.sample_rate))
        right_channel = np.zeros(int(duration * self.sample_rate))
        
        if len(points) == 0:
            return np.column_stack((left_channel, right_channel))
        
        # Create spatial grid that includes ALL points
        grid_resolution = int(np.sqrt(len(points) / 100))  # Adaptive grid size
        grid_resolution = max(10, min(grid_resolution, 50))  # Reasonable bounds
        
        # Safe coordinate ranges
        points_real = np.real(points)
        x_coords = points_real[:, 0]
        y_coords = points_real[:, 1]
        
        x_finite = x_coords[np.isfinite(x_coords)]
        y_finite = y_coords[np.isfinite(y_coords)]
        
        if len(x_finite) == 0 or len(y_finite) == 0:
            return np.column_stack((left_channel, right_channel))
        
        # Divide space into grid
        x_bins = np.linspace(np.min(x_finite), np.max(x_finite), grid_resolution + 1)
        y_bins = np.linspace(np.min(y_finite), np.max(y_finite), grid_resolution + 1)
        
        # Process each grid cell
        cell_duration = duration / (grid_resolution * grid_resolution)
        
        cell_count = 0
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Find ALL points in this cell
                cell_mask = (
                    (x_coords >= x_bins[i]) & (x_coords < x_bins[i + 1]) &
                    (y_coords >= y_bins[j]) & (y_coords < y_bins[j + 1])
                )
                
                cell_points = points[cell_mask]
                if len(cell_points) == 0:
                    cell_count += 1
                    continue
                
                cell_clusters = clusters[cell_mask]
                cell_anomalies = anomalies[cell_mask]
                
                # Process ALL points in this cell
                self._render_full_cell_audio_safe(
                    cell_points, cell_clusters, cell_anomalies,
                    cell_count * cell_duration, cell_duration,
                    left_channel, right_channel, points
                )
                
                cell_count += 1
        
        # Safe normalization
        left_channel = self._ensure_real_audio(left_channel)
        right_channel = self._ensure_real_audio(right_channel)
        
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel /= max_val * 1.1
            right_channel /= max_val * 1.1
        
        return np.column_stack((left_channel, right_channel))
    
    def _render_full_cell_audio_safe(self, cell_points, cell_clusters, cell_anomalies,
                                   start_time, cell_duration, left_channel, right_channel, all_points):
        """Render audio for ALL points in a cell - with comprehensive safety checks"""
        
        if len(cell_points) == 0:
            return
        
        try:
            # Safe calculations
            elevations = np.real(cell_points[:, 2])
            elevations = elevations[np.isfinite(elevations)]
            
            if len(elevations) == 0:
                return
            
            avg_elevation = np.mean(elevations)
            elevation_spread = np.std(elevations)
            
            # Safe frequency calculation
            all_elevations = np.real(all_points[:, 2])
            all_elevations = all_elevations[np.isfinite(all_elevations)]
            
            if len(all_elevations) == 0:
                base_frequency = 440.0
            else:
                elev_range = np.ptp(all_elevations)
                if elev_range == 0:
                    base_frequency = 440.0
                else:
                    elev_norm = (avg_elevation - np.min(all_elevations)) / elev_range
                    elev_norm = np.clip(elev_norm, 0, 1)
                    base_frequency = 250 + elev_norm * 600
            
            base_frequency = np.clip(base_frequency, 80, 2000)
            
            # Determine dominant cluster safely
            dominant_cluster = 0
            unique_clusters, counts = np.unique(cell_clusters[cell_clusters >= 0], return_counts=True)
            if len(unique_clusters) > 0:
                dominant_cluster = unique_clusters[np.argmax(counts)]
            
            # Generate audio using synthesizer
            audio_signal = self.synthesizer.synthesize_cluster_tone(
                float(base_frequency), float(cell_duration), int(dominant_cluster), 0.3
            )
            audio_signal = self._ensure_real_audio(audio_signal)
            
            # Modify based on terrain characteristics
            if len(elevations) > 1 and elevation_spread > 0:
                try:
                    texture_audio = self.synthesizer.surface_to_audio_texture(
                        'curved', float(base_frequency), float(cell_duration), 0.2
                    )
                    texture_audio = self._ensure_real_audio(texture_audio)
                    min_len = min(len(audio_signal), len(texture_audio))
                    if min_len > 0:
                        audio_signal[:min_len] += texture_audio[:min_len] * 0.5
                except Exception:
                    pass  # Skip texture if it fails
            
            # Point density influence
            point_density = len(cell_points)
            if point_density > 0:
                density_factor = np.log10(point_density + 1) / np.log10(1000)
                density_factor = np.clip(density_factor, 0.1, 1.0)
                audio_signal *= (0.5 + density_factor * 0.5)
            
            # Anomaly modification
            anomaly_ratio = np.sum(cell_anomalies) / max(len(cell_points), 1)
            if anomaly_ratio > 0.1:
                anomaly_factor = 1 + anomaly_ratio
                anomaly_factor = np.clip(anomaly_factor, 1.0, 2.0)
                audio_signal *= anomaly_factor
            
            # Safe HRTF positioning
            cell_center = np.mean(np.real(cell_points), axis=0)
            cell_center = cell_center[np.isfinite(cell_center)]
            
            if len(cell_center) >= 3:
                positioned_audio = self.hrtf_processor.apply_hrtf_positioning(audio_signal, cell_center)
                positioned_audio = self._ensure_real_audio(positioned_audio)
            else:
                # Fallback to simple stereo positioning
                positioned_audio = np.column_stack((audio_signal, audio_signal))
            
            # Safe mixing
            start_sample = int(start_time * self.sample_rate)
            end_sample = min(start_sample + len(positioned_audio), len(left_channel))
            audio_length = end_sample - start_sample
            
            if audio_length > 0 and start_sample >= 0:
                left_channel[start_sample:end_sample] += positioned_audio[:audio_length, 0]
                right_channel[start_sample:end_sample] += positioned_audio[:audio_length, 1]
                
        except Exception as e:
            # Silently skip problematic cells
            return

    # ============ MODE 3: ADVANCED - FULL MULTITRACK WITH ALL POINTS ============
    def generate_full_featured_sonification(self, analysis_result, tempo=120, duration=30):
        """MODE 3: Full multitrack using ALL points through sophisticated components"""
        
        try:
            points = analysis_result['points']
            clusters = analysis_result['clusters']
            anomalies = analysis_result['anomalies']
            surface_types = analysis_result.get('surface_types', ['unknown'] * len(points))
            geometric_features = analysis_result.get('geometric_features', {})
            
            # Use MultiTrackRenderer with ALL points - it handles the complexity
            self.multitrack_renderer.clear_tracks()
            elevations = points[:, 2]
            
            # The MultiTrackRenderer is designed to handle large datasets efficiently
            self.multitrack_renderer.add_background_track(points, surface_types, elevations, duration)
            self.multitrack_renderer.add_cluster_tracks(points, clusters, elevations, duration)
            self.multitrack_renderer.add_feature_events(geometric_features, points, duration)
            self.multitrack_renderer.add_anomaly_alerts(points, anomalies, elevations, duration)
            
            # Render using ALL the sophisticated processing
            advanced_audio = self.multitrack_renderer.render_multitrack_audio(duration)
            advanced_audio = self._ensure_real_audio(advanced_audio)
            
            # Ensure stereo format
            if advanced_audio.ndim == 1:
                advanced_audio = np.column_stack((advanced_audio, advanced_audio))
            
            wav_filename = self.save_wav(advanced_audio, "advanced_sonification.wav")
            midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration, downsample_factor=max(1, len(points)//2000))
            midi_filename = self.save_midi(midi_data, "advanced_sonification.mid")
            
            return {
                'wav_file': wav_filename,
                'midi_file': midi_filename,
                'audio_data': advanced_audio
            }
        except Exception as e:
            raise ValueError(f"Advanced sonification failed: {str(e)}")

    # ============ SHARED METHODS ============
    def generate_midi_sonification(self, points, clusters, anomalies, tempo=120, duration=30, downsample_factor=20):
        if len(points) == 0:
            raise ValueError("No points to sonify")
        
        midi_file = MIDIFile(4)
        for track in range(4):
            midi_file.addTempo(track, 0, tempo)
        
        # Reasonable downsampling for MIDI (MIDI can't handle millions of notes)
        indices = np.arange(0, len(points), downsample_factor)
        selected_points = points[indices]
        selected_clusters = clusters[indices] 
        selected_anomalies = anomalies[indices]
        
        # Safe elevation processing
        elevations = np.real(selected_points[:, 2])
        elevations = elevations[np.isfinite(elevations)]
        
        if len(elevations) == 0:
            return midi_file
        
        normalized = self._safe_normalize(elevations)
        note_indices = (normalized * (len(self.scale_notes) - 1)).astype(int)
        note_indices = np.clip(note_indices, 0, len(self.scale_notes) - 1)
        midi_notes = [self.scale_notes[idx] for idx in note_indices]
        
        beats_per_second = tempo / 60
        total_beats = duration * beats_per_second
        beat_step = total_beats / len(selected_points)
        
        for i in range(min(len(selected_points), len(midi_notes))):
            beat_time = i * beat_step
            note = midi_notes[i] if i < len(midi_notes) else 60
            is_anomaly = selected_anomalies[i] if i < len(selected_anomalies) else False
            cluster_id = selected_clusters[i] if (i < len(selected_clusters) and selected_clusters[i] >= 0) else 0
            
            track = min(int(cluster_id) % 3, 2)
            if is_anomaly:
                track = 3
            
            channel = min(int(cluster_id) % 16, 15)
            duration_beats = min(beat_step, 0.5)
            velocity = 100 if is_anomaly else 70
            
            midi_file.addNote(track, channel, int(note), beat_time, duration_beats, int(velocity))
            
            if is_anomaly:
                midi_file.addNote(3, 9, 42, beat_time, 0.1, 120)
        
        return midi_file
    
    def save_wav(self, audio_data, filename="sonification.wav"):
        # Ensure audio is real and properly formatted
        audio_data = self._ensure_real_audio(audio_data)
        
        # Ensure stereo format
        if audio_data.ndim == 1:
            audio_data = np.column_stack((audio_data, audio_data))
        
        # Convert to 16-bit integer
        audio_int = (audio_data * 32767).astype(np.int16)
        wav.write(filename, self.sample_rate, audio_int)
        return filename
    
    def save_midi(self, midi_file, filename="sonification.mid"):
        with open(filename, "wb") as output_file:
            midi_file.writeFile(output_file)
        return filename
