import cv2
import numpy as np
import librosa

def solution(audio_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Parameters for spectrogram analysis
        n_fft = 2048
        hop_length = n_fft // 2

        # Calculate the spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)
        spec_mean=np.mean(spec)
        threshold = 1 # set the threshold 

        # Determine the quality based on the threshold
        return 'metal' if spec_mean >= threshold else 'cardboard'
    
    except Exception as e:
        return str(e)


   
