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
        
    def normalize_data(self, data):
        if np.ptp(data) == 0:
            return np.zeros_like(data)
        return (data - data.min()) / np.ptp(data)
    
    def elevation_to_frequency(self, elevations):
        normalized = self.normalize_data(elevations)
        freq_indices = (normalized * (len(self.scale_freqs) - 1)).astype(int)
        return [self.scale_freqs[idx] for idx in freq_indices]
    
    def elevation_to_midi_note(self, elevations):
        normalized = self.normalize_data(elevations)
        note_indices = (normalized * (len(self.scale_notes) - 1)).astype(int)
        return [self.scale_notes[idx] for idx in note_indices]
    
    def generate_advanced_sonification(self, analysis_result, tempo=120, duration=30):
        points = analysis_result['points']
        clusters = analysis_result['clusters']
        anomalies = analysis_result['anomalies']
        surface_types = analysis_result['surface_types']
        geometric_features = analysis_result['geometric_features']
        normals = analysis_result.get('normals', [])
        
        elevations = points[:, 2]
        
        semantic_labels = None
        try:
            semantic_labels = self.semantic_labeler.apply_semantic_labels(
                points, clusters, surface_types
            )
        except:
            pass
        
        self.multitrack_renderer.clear_tracks()
        
        self.multitrack_renderer.add_background_track(points, surface_types, elevations, duration)
        self.multitrack_renderer.add_cluster_tracks(points, clusters, elevations, duration)
        self.multitrack_renderer.add_feature_events(geometric_features, points, duration)
        self.multitrack_renderer.add_anomaly_alerts(points, anomalies, elevations, duration)
        
        audio_data = self.multitrack_renderer.render_multitrack_audio(duration)
        
        return audio_data
    
    def generate_wav_sonification(self, points, clusters, anomalies, 
                                tempo=120, duration=30, downsample_factor=20):
        
        if len(points) == 0:
            raise ValueError("No points to sonify")
        
        indices = np.arange(0, len(points), downsample_factor)
        selected_points = points[indices]
        selected_clusters = clusters[indices] 
        selected_anomalies = anomalies[indices]
        
        elevation_order = np.argsort(selected_points[:, 2])
        
        frequencies = self.elevation_to_frequency(selected_points[:, 2])
        
        left_channel = np.zeros(int(duration * self.sample_rate))
        right_channel = np.zeros(int(duration * self.sample_rate))
        
        note_duration = 0.3
        time_step = duration / len(elevation_order)
        
        for i, idx in enumerate(elevation_order):
            time_pos = i * time_step
            sample_pos = int(time_pos * self.sample_rate)
            
            if sample_pos >= len(left_channel) - int(note_duration * self.sample_rate):
                break
                
            frequency = frequencies[idx]
            is_anomaly = selected_anomalies[idx]
            cluster_id = selected_clusters[idx] if selected_clusters[idx] >= 0 else 0
            
            amplitude = 0.5 if is_anomaly else 0.3
            
            tone = self.synthesizer.synthesize_cluster_tone(
                frequency, note_duration, cluster_id, amplitude
            )
            
            stereo_tone = self.hrtf_processor.apply_hrtf_positioning(
                tone, selected_points[idx]
            )
            
            end_sample = min(sample_pos + len(stereo_tone), len(left_channel))
            tone_length = end_sample - sample_pos
            
            if tone_length > 0:
                left_channel[sample_pos:end_sample] += stereo_tone[:tone_length, 0]
                right_channel[sample_pos:end_sample] += stereo_tone[:tone_length, 1]
        
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel = left_channel / max_val * 0.8
            right_channel = right_channel / max_val * 0.8
        
        stereo_audio = np.column_stack((left_channel, right_channel))
        return stereo_audio
    
    def generate_midi_sonification(self, points, clusters, anomalies, 
                                 tempo=120, duration=30, downsample_factor=20):
        
        if len(points) == 0:
            raise ValueError("No points to sonify")
        
        midi_file = MIDIFile(4)
        for track in range(4):
            midi_file.addTempo(track, 0, tempo)
        
        indices = np.arange(0, len(points), downsample_factor)
        selected_points = points[indices]
        selected_clusters = clusters[indices] 
        selected_anomalies = anomalies[indices]
        
        elevation_order = np.argsort(selected_points[:, 2])
        
        midi_notes = self.elevation_to_midi_note(selected_points[:, 2])
        
        beats_per_second = tempo / 60
        total_beats = duration * beats_per_second
        beat_step = total_beats / len(elevation_order)
        
        for i, idx in enumerate(elevation_order):
            beat_time = i * beat_step
            note = midi_notes[idx]
            is_anomaly = selected_anomalies[idx]
            cluster_id = selected_clusters[idx] if selected_clusters[idx] >= 0 else 0
            
            track = min(cluster_id % 3, 2)
            if is_anomaly:
                track = 3
            
            channel = min(cluster_id % 16, 15) if cluster_id >= 0 else 9
            
            duration_beats = min(beat_step, 0.5)
            velocity = 100 if is_anomaly else 70
            
            midi_file.addNote(track, channel, note, beat_time, duration_beats, velocity)
            
            if is_anomaly:
                midi_file.addNote(3, 9, 42, beat_time, 0.1, 120)
        
        return midi_file
    
    def save_wav(self, audio_data, filename="sonification.wav"):
        audio_int = (audio_data * 32767).astype(np.int16)
        wav.write(filename, self.sample_rate, audio_int)
        return filename
    
    def save_midi(self, midi_file, filename="sonification.mid"):
        with open(filename, "wb") as output_file:
            midi_file.writeFile(output_file)
        return filename
    
    def generate_clean_sonification(self, points, clusters, anomalies, tempo=120, duration=20):
        audio_data = self.generate_wav_sonification(points, clusters, anomalies, tempo, duration)
        wav_filename = self.save_wav(audio_data, "clean_sonification.wav")
        
        midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration)
        midi_filename = self.save_midi(midi_data, "clean_sonification.mid")
        
        return {
            'wav_file': wav_filename,
            'midi_file': midi_filename,
            'audio_data': audio_data
        }
    
    def generate_spatial_sonification(self, points, clusters, anomalies, tempo=100, duration=30):
        audio_data = self.generate_wav_sonification(points, clusters, anomalies, tempo, duration)
        wav_filename = self.save_wav(audio_data, "spatial_sonification.wav")
        
        midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration)
        midi_filename = self.save_midi(midi_data, "spatial_sonification.mid")
        
        return {
            'wav_file': wav_filename,
            'midi_file': midi_filename,
            'audio_data': audio_data
        }
    
    def generate_full_featured_sonification(self, analysis_result, tempo=120, duration=30):
        advanced_audio = self.generate_advanced_sonification(analysis_result, tempo, duration)
        wav_filename = self.save_wav(advanced_audio, "advanced_sonification.wav")
        
        points = analysis_result['points']
        clusters = analysis_result['clusters']
        anomalies = analysis_result['anomalies']
        
        midi_data = self.generate_midi_sonification(points, clusters, anomalies, tempo, duration)
        midi_filename = self.save_midi(midi_data, "advanced_sonification.mid")
        
        return {
            'wav_file': wav_filename,
            'midi_file': midi_filename,
            'audio_data': advanced_audio
        }
