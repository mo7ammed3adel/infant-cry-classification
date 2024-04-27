import librosa
import numpy as np
import soundfile as sf


def audio_to_spectrogram(file_path, target_size=(128, 128)):
    # Load audio file
    data, sample_rate = sf.read(file_path)
    
    # Generate a Mel-spectrogram
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Resize the spectrogram to the target size
    import cv2
    S_DB_resized = cv2.resize(S_DB, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize the spectrogram
    S_DB_normalized = (S_DB_resized - np.min(S_DB_resized)) / (np.max(S_DB_resized) - np.min(S_DB_resized))
    
    # Add a channel dimension
    S_DB_normalized = np.expand_dims(S_DB_normalized, axis=-1)
    
    return S_DB_normalized

