import numpy as np
from instrument_synthesizer import InstrumentSynthesizer
from hrtf_processor import HRTFProcessor

class MultiTrackRenderer:
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.synthesizer = InstrumentSynthesizer(sample_rate)
        self.hrtf_processor = HRTFProcessor()
        self.tracks = {
            'background': [],
            'clusters': [],
            'features': [],
            'anomalies': []
        }
        self.track_volumes = {
            'background': 0.3,
            'clusters': 0.7,
            'features': 0.5,
            'anomalies': 0.8
        }
    
    def clear_tracks(self):
        for track in self.tracks:
            self.tracks[track] = []
    
    def add_background_track(self, points, surface_types, elevations, duration=30):
        selected_indices = np.linspace(0, len(points)-1, min(len(points), 200), dtype=int)
        
        for idx in selected_indices:
            point = points[idx]
            surface_type = surface_types[idx]
            elevation = elevations[idx]
            
            frequency = self._elevation_to_frequency(elevation, elevations)
            note_duration = 0.2
            start_time = (idx / len(selected_indices)) * duration
            
            audio_signal = self.synthesizer.surface_to_audio_texture(
                surface_type, frequency, note_duration, amplitude=0.2
            )
            
            self.tracks['background'].append({
                'audio': audio_signal,
                'start_time': start_time,
                'position': point,
                'duration': note_duration
            })
    
    def add_cluster_tracks(self, points, clusters, elevations, duration=30):
        unique_clusters = np.unique(clusters[clusters != -1])
        
        for cluster_id in unique_clusters:
            cluster_points = points[clusters == cluster_id]
            cluster_elevations = elevations[clusters == cluster_id]
            
            selected_indices = np.linspace(0, len(cluster_points)-1, 
                                         min(len(cluster_points), 50), dtype=int)
            
            for i, idx in enumerate(selected_indices):
                point = cluster_points[idx]
                elevation = cluster_elevations[idx]
                
                frequency = self._elevation_to_frequency(elevation, elevations)
                note_duration = 0.4
                start_time = (i / len(selected_indices)) * duration * 0.8
                
                audio_signal = self.synthesizer.synthesize_cluster_tone(
                    frequency, note_duration, cluster_id, amplitude=0.4
                )
                
                self.tracks['clusters'].append({
                    'audio': audio_signal,
                    'start_time': start_time,
                    'position': point,
                    'duration': note_duration,
                    'cluster_id': cluster_id
                })
    
    def add_feature_events(self, geometric_features, points, duration=30):
        all_features = []
        
        for feature_type, feature_list in geometric_features.items():
            for feature in feature_list:
                all_features.append((feature_type, feature))
        
        for i, (feature_type, feature) in enumerate(all_features):
            point = feature['position']
            start_time = (i / max(len(all_features), 1)) * duration
            
            if feature_type == 'planes':
                audio_signal = self._create_plane_chord(point)
            elif feature_type == 'edges':
                audio_signal = self._create_edge_ping(point)
            elif feature_type == 'corners':
                audio_signal = self._create_corner_sound(point)
            else:
                continue
            
            self.tracks['features'].append({
                'audio': audio_signal,
                'start_time': start_time,
                'position': point,
                'duration': 0.3,
                'feature_type': feature_type
            })
    
    def add_anomaly_alerts(self, points, anomalies, elevations, duration=30):
        anomaly_points = points[anomalies]
        anomaly_elevations = elevations[anomalies]
        
        for i, (point, elevation) in enumerate(zip(anomaly_points, anomaly_elevations)):
            frequency = self._elevation_to_frequency(elevation, elevations) * 2
            start_time = (i / max(len(anomaly_points), 1)) * duration
            
            audio_signal = self._create_anomaly_alert(frequency)
            
            self.tracks['anomalies'].append({
                'audio': audio_signal,
                'start_time': start_time,
                'position': point,
                'duration': 0.2
            })
    
    def _elevation_to_frequency(self, elevation, all_elevations):
        if len(all_elevations) == 0 or np.ptp(all_elevations) == 0:
            return 440.0
        
        normalized = (elevation - np.min(all_elevations)) / np.ptp(all_elevations)
        scale_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        freq_index = int(normalized * (len(scale_notes) - 1))
        return scale_notes[freq_index]
    
    def _create_plane_chord(self, point):
        base_freq = 220
        chord_frequencies = [base_freq, base_freq * 1.25, base_freq * 1.5]
        duration = 0.5
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.zeros_like(t)
        
        for freq in chord_frequencies:
            wave += np.sin(2 * np.pi * freq * t) / len(chord_frequencies)
        
        envelope = np.exp(-t * 2)
        return wave * envelope * 0.3
    
    def _create_edge_ping(self, point):
        frequency = 880
        duration = 0.1
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        
        envelope = np.exp(-t * 20)
        return wave * envelope * 0.4
    
    def _create_corner_sound(self, point):
        frequency = 660
        duration = 0.3
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        for i in range(2, 5):
            wave += (0.5 / i) * np.sin(2 * np.pi * frequency * i * t)
        
        envelope = np.exp(-t * 5)
        return wave * envelope * 0.35
    
    def _create_anomaly_alert(self, frequency):
        duration = 0.2
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        wave += np.sin(2 * np.pi * frequency * 1.5 * t)
        
        envelope = np.exp(-t * 8)
        return wave * envelope * 0.6
    
    def _get_track_volume(self, track_name):
        return self.track_volumes.get(track_name, 0.5)
    
    def render_multitrack_audio(self, duration):
        total_samples = int(duration * self.sample_rate)
        final_audio = np.zeros((total_samples, 2))
        
        for track_name, events in self.tracks.items():
            track_volume = self._get_track_volume(track_name)
            
            for event in events:
                start_sample = int(event['start_time'] * self.sample_rate)
                audio_signal = event['audio']
                position = event['position']
                
                if len(audio_signal) == 0:
                    continue
                
                stereo_audio = self.hrtf_processor.apply_hrtf_positioning(
                    audio_signal, position
                )
                
                end_sample = min(start_sample + len(stereo_audio), total_samples)
                audio_length = end_sample - start_sample
                
                if audio_length > 0:
                    final_audio[start_sample:end_sample] += stereo_audio[:audio_length] * track_volume
        
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.8
        
        return final_audio
