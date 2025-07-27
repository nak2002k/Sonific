import numpy as np
from scipy import signal

class InstrumentSynthesizer:
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.instruments = {
            'piano': self._piano_synthesis,
            'violin': self._violin_synthesis,
            'flute': self._flute_synthesis,
            'drums': self._drum_synthesis,
            'marimba': self._marimba_synthesis,
            'bell': self._bell_synthesis,
            'pad': self._pad_synthesis
        }
    
    def get_cluster_instrument(self, cluster_id):
        instrument_list = list(self.instruments.keys())
        return instrument_list[cluster_id % len(instrument_list)]
    
    def synthesize_cluster_tone(self, frequency, duration, cluster_id, amplitude=0.3):
        instrument = self.get_cluster_instrument(cluster_id)
        return self.instruments[instrument](frequency, duration, amplitude)
    
    def _piano_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        harmonics = [1, 0.5, 0.25, 0.125, 0.0625]
        wave = np.zeros_like(t)
        
        for i, harmonic_amp in enumerate(harmonics):
            wave += harmonic_amp * np.sin(2 * np.pi * frequency * (i + 1) * t)
        
        envelope = np.exp(-t * 3)
        wave *= envelope * amplitude
        
        return wave
    
    def _violin_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        sawtooth = signal.sawtooth(2 * np.pi * frequency * t)
        
        cutoff_freq = frequency * 4
        b, a = signal.butter(4, cutoff_freq / (self.sample_rate / 2), 'low')
        filtered_wave = signal.lfilter(b, a, sawtooth)
        
        attack_time = 0.1
        release_time = 0.3
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        envelope = np.ones_like(t)
        if len(envelope) > attack_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if len(envelope) > release_samples:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        return filtered_wave * envelope * amplitude
    
    def _flute_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        wave = np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        wave += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
        
        breath_noise = np.random.normal(0, 0.05, len(t))
        wave += breath_noise
        
        cutoff_freq = frequency * 6
        b, a = signal.butter(2, cutoff_freq / (self.sample_rate / 2), 'low')
        wave = signal.lfilter(b, a, wave)
        
        envelope = np.exp(-t * 1.5)
        return wave * envelope * amplitude
    
    def _drum_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        base_freq = max(frequency * 0.5, 40)
        wave = np.sin(2 * np.pi * base_freq * t)
        
        noise = np.random.normal(0, 0.3, len(t))
        wave += noise
        
        envelope = np.exp(-t * 8)
        return wave * envelope * amplitude
    
    def _marimba_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        harmonics = [1, 0.3, 0.1, 0.05]
        wave = np.zeros_like(t)
        
        for i, harmonic_amp in enumerate(harmonics):
            wave += harmonic_amp * np.sin(2 * np.pi * frequency * (i + 1) * t)
        
        envelope = np.exp(-t * 4)
        metallic_ring = 0.1 * np.sin(2 * np.pi * frequency * 3.14 * t) * np.exp(-t * 6)
        wave += metallic_ring
        
        return wave * envelope * amplitude
    
    def _bell_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        harmonics = [1, 1.67, 2.33, 3.14, 4.27]
        harmonic_amps = [1, 0.6, 0.4, 0.3, 0.2]
        wave = np.zeros_like(t)
        
        for harmonic_freq, harmonic_amp in zip(harmonics, harmonic_amps):
            wave += harmonic_amp * np.sin(2 * np.pi * frequency * harmonic_freq * t)
        
        envelope = np.exp(-t * 2)
        return wave * envelope * amplitude
    
    def _pad_synthesis(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        wave1 = np.sin(2 * np.pi * frequency * t)
        wave2 = np.sin(2 * np.pi * frequency * 1.01 * t)
        wave3 = np.sin(2 * np.pi * frequency * 0.99 * t)
        
        wave = (wave1 + 0.7 * wave2 + 0.7 * wave3) / 2.4
        
        cutoff_freq = frequency * 3
        b, a = signal.butter(2, cutoff_freq / (self.sample_rate / 2), 'low')
        wave = signal.lfilter(b, a, wave)
        
        envelope = 1 - np.exp(-t * 5)
        return wave * envelope * amplitude
    
    def surface_to_audio_texture(self, surface_type, frequency, duration, amplitude=0.3):
        if surface_type == 'flat':
            return self._generate_harmonic_chord(frequency, duration, amplitude)
        elif surface_type == 'edge':
            return self._generate_sharp_ping(frequency * 2, duration * 0.3, amplitude)
        elif surface_type == 'curved':
            return self._generate_modulated_tone(frequency, duration, amplitude)
        else:
            return self._piano_synthesis(frequency, duration, amplitude)
    
    def _generate_harmonic_chord(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        chord_frequencies = [frequency, frequency * 1.25, frequency * 1.5]
        wave = np.zeros_like(t)
        
        for freq in chord_frequencies:
            wave += np.sin(2 * np.pi * freq * t) / len(chord_frequencies)
        
        envelope = np.exp(-t * 1.5)
        return wave * envelope * amplitude
    
    def _generate_sharp_ping(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        wave = np.sin(2 * np.pi * frequency * t)
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        
        envelope = np.exp(-t * 15)
        return wave * envelope * amplitude
    
    def _generate_modulated_tone(self, frequency, duration, amplitude):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        modulation_rate = 5
        modulation_depth = 0.1
        
        modulated_freq = frequency * (1 + modulation_depth * np.sin(2 * np.pi * modulation_rate * t))
        
        phase = np.cumsum(2 * np.pi * modulated_freq / self.sample_rate)
        wave = np.sin(phase)
        
        envelope = np.exp(-t * 2)
        return wave * envelope * amplitude
