import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

def generate_spectrogram(audio_path, output_path=None, dpi=250, cmap='viridis', title='Spectrogram of Sonification'):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(9, 4))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap=cmap)
    plt.title(title)
    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            output_path = tmp.name
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
