from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import glob
import os
import random

def get_wav_duration(file_path):
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)
    print(sample_rate)
    print(len(data), data.dtype)
    # Calculate duration
    duration = len(data) / float(sample_rate)
    # plt.plot(np.arange(0, data.shape[-1], 1), data)
    # plt.show()

    return duration, sample_rate, data

folder_path = '/Users/Omer/Documents/DirectPhD/BSA/Final project/Signals/Segmented_signals'

# Get a list of all files in the folder matching a specific pattern
folders = glob.glob(os.path.join(folder_path, '*'))
folders_to_iter = random.sample(folders, 10)

# Loop over each file
for folder in folders_to_iter:
    print(folder)
    files = glob.glob(os.path.join(folder, '*'))

    x = folder.rfind('/')+1
    fig, axes = plt.subplots(len(files), 5, figsize=(18,10))
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    for i, file in enumerate(files):

        duration, sample_rate, data = get_wav_duration(file)            
        dt = 1/sample_rate # difference from one time point to the next
        ax = axes[i, 0]
        ax.plot(np.arange(0, data.shape[-1]*dt, dt), data)
        ax.set_ylabel('Voltage (V)')
        axes[0, 0].set_title("Time domain")#, segment {}".format(i))
    
        sigma = 50
        smoothed_curve = gaussian_filter(data, sigma=sigma)
        ax = axes[i, 1]
        ax.plot(np.arange(0, data.shape[-1]*dt, dt), smoothed_curve)
        ax.set_ylabel('Voltage (V)')
        axes[0, 1].set_title("Time domain (filtered)")

        N = len(data) # number of samples
        T = N*dt # total time sample -> T = NΔ
        # print(T)
        data_transformed = np.fft.rfft(data)
        spectrum = (2 * dt ** 2 / T * data_transformed  * data_transformed.conj()).real
        faxis = np.arange(len(spectrum)) / T
        decibel_scaled_data = 10 * np.log10(spectrum / max(spectrum))
        
        # print("faxis size is:", len(faxis))
        ax = axes[i, 2]
        ax.plot(faxis[:100], decibel_scaled_data[:100])
        ax.set_ylabel('dB')
        axes[0, 2].set_title("Freq domain scaled")#, segment {}".format(i))

        ax = axes[i, 3]
        ax.plot(faxis[:100], spectrum[:100])
        ax.set_ylabel('dB')
        axes[0, 3].set_title("Freq domain NOT scaled")#, segment {}".format(i))

        ax = axes[i, 4]
        # Compute spectrogram
        spectrum, frequencies, times, image = ax.specgram(data, NFFT=1024, Fs=sample_rate)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 2000)
        axes[0, 4].set_title("Spectogram")#, segment {}".format(i))

        axes[-1,0].set_xlabel('Time (seconds)')
        axes[-1,1].set_xlabel('Time (seconds)')
        axes[-1,2].set_xlabel('Freq (Hz)')
        axes[-1,3].set_xlabel('Freq (Hz)')
        axes[-1,4].set_xlabel('Time (s)')
            
    fig.suptitle(folder)
    plt.show()
    # print(files)
    
# # Example usage
# file_path = '/Users/Omer/Downloads/Segmented_signals/BMR2_vs_BMR3_101/segment_5.wav'  # Replace 'example.wav' with the path to your WAV file
# duration = get_wav_duration(file_path)
# print("Duration of the WAV file:", duration, "seconds")

# # Example 2 usage
# file_path = '/Users/Omer/Downloads/Segmented_signals/BMR2_vs_BMR3_128/segment_2.wav'  # Replace 'example.wav' with the path to your WAV file
# duration = get_wav_duration(file_path)
# print("Duration of the WAV file:", duration, "seconds")

