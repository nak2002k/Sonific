import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist

class HRTFProcessor:
    
    def __init__(self):
        self.sample_rate = 44100
        self.hrtf_data = self._generate_simplified_hrtf()
    
    def _generate_simplified_hrtf(self):
        azimuths = np.arange(0, 360, 15)
        elevations = np.arange(-40, 91, 10)
        
        hrtf_data = {}
        
        for azimuth in azimuths:
            for elevation in elevations:
                left_filter, right_filter = self._create_hrtf_filters(azimuth, elevation)
                hrtf_data[(azimuth, elevation)] = {
                    'left': left_filter,
                    'right': right_filter
                }
        
        return hrtf_data
    
    def _create_hrtf_filters(self, azimuth, elevation):
        filter_length = 128
        
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        itd_samples = int(0.0006 * np.sin(azimuth_rad) * self.sample_rate)
        
        left_delay = max(0, -itd_samples)
        right_delay = max(0, itd_samples)
        
        head_shadow_freq = 3000 + 2000 * abs(np.sin(azimuth_rad))
        head_shadow_freq = np.clip(head_shadow_freq, 100, self.sample_rate / 2 - 100)
        
        b, a = signal.butter(2, head_shadow_freq / (self.sample_rate / 2), 'low')
        impulse = signal.unit_impulse(filter_length)
        base_response = signal.lfilter(b, a, impulse)
        
        # Ensure real values
        base_response = np.real(base_response)
        
        left_filter = np.zeros(filter_length + left_delay)
        right_filter = np.zeros(filter_length + right_delay)
        
        left_amplitude = 1.0 - 0.3 * max(0, np.cos(azimuth_rad))
        right_amplitude = 1.0 + 0.3 * max(0, np.cos(azimuth_rad))
        
        elevation_gain = 1.0 + 0.2 * np.sin(elevation_rad)
        
        left_filter[left_delay:left_delay + filter_length] = base_response * left_amplitude * elevation_gain
        right_filter[right_delay:right_delay + filter_length] = base_response * right_amplitude * elevation_gain
        
        # Ensure filters are real
        left_filter = np.real(left_filter)
        right_filter = np.real(right_filter)
        
        return left_filter, right_filter
    
    def _find_nearest_hrtf(self, target_azimuth, target_elevation):
        min_distance = float('inf')
        nearest_key = None
        
        for (azimuth, elevation) in self.hrtf_data.keys():
            distance = np.sqrt((azimuth - target_azimuth)**2 + (elevation - target_elevation)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_key = (azimuth, elevation)
        
        return nearest_key
    
    def calculate_position_angles(self, point_position, listener_position=None):
        if listener_position is None:
            listener_position = np.array([0, 0, 0])
        
        # Ensure real coordinates
        point_position = np.real(point_position)
        listener_position = np.real(listener_position)
        
        relative_position = point_position - listener_position
        
        azimuth = np.degrees(np.arctan2(relative_position[1], relative_position[0]))
        if azimuth < 0:
            azimuth += 360
        
        horizontal_distance = np.sqrt(relative_position[0]**2 + relative_position[1]**2)
        if horizontal_distance == 0:
            elevation = 90 if relative_position[2] > 0 else -90
        else:
            elevation = np.degrees(np.arctan2(relative_position[2], horizontal_distance))
        
        elevation = np.clip(elevation, -40, 90)
        
        return float(azimuth), float(elevation)
    
    def apply_hrtf_positioning(self, audio_signal, point_position, listener_position=None):
        if len(audio_signal) == 0:
            return np.zeros((1, 2))
        
        # Ensure real audio
        audio_signal = np.real(audio_signal)
        audio_signal = np.nan_to_num(audio_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            azimuth, elevation = self.calculate_position_angles(point_position, listener_position)
            
            nearest_key = self._find_nearest_hrtf(azimuth, elevation)
            hrtf_filters = self.hrtf_data[nearest_key]
            
            left_filter = hrtf_filters['left']
            right_filter = hrtf_filters['right']
            
            left_channel = signal.convolve(audio_signal, left_filter, mode='full')
            right_channel = signal.convolve(audio_signal, right_filter, mode='full')
            
            # Ensure real results
            left_channel = np.real(left_channel)
            right_channel = np.real(right_channel)
            
            max_length = max(len(left_channel), len(right_channel))
            
            if len(left_channel) < max_length:
                left_channel = np.pad(left_channel, (0, max_length - len(left_channel)))
            if len(right_channel) < max_length:
                right_channel = np.pad(right_channel, (0, max_length - len(right_channel)))
            
            return np.column_stack([left_channel, right_channel])
            
        except Exception as e:
            # Fallback: simple stereo positioning
            x_norm = 0.5  # Center position
            if len(point_position) >= 1:
                x_norm = np.clip((point_position[0] + 1000) / 2000, 0, 1)  # Rough normalization
            
            left_gain = 1 - x_norm
            right_gain = x_norm
            
            return np.column_stack([audio_signal * left_gain, audio_signal * right_gain])
    
    def apply_distance_attenuation(self, audio_signal, distance):
        if distance <= 0:
            distance = 1.0
        
        attenuation_factor = 1.0 / (1.0 + distance * 0.001)
        return np.real(audio_signal * attenuation_factor)
