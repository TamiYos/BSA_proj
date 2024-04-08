from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import glob
import os

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

folder_path = '/Users/Omer/Downloads/Segmented_signals/'

# Get a list of all files in the folder matching a specific pattern
folders = glob.glob(os.path.join(folder_path, '*'))

# Loop over each file
for folder in folders:
    print(folder)
    files = glob.glob(os.path.join(folder, '*'))
    # print(folder[folder.rfind('/'):folder.rfind('/')+5])
    x = folder.rfind('/')+1
    if folder[x:x+4] == "BMR2":
        # print('BMR2')
        fig, axes = plt.subplots(len(files), 4, figsize=(15,10))
    for i, file in enumerate(files):
        if folder[x:x+4] == "BMR2":
            duration, sample_rate, data = get_wav_duration(file)
            dt = 1/sample_rate # difference from one time point to the next
            ax = axes[i, 0]
            ax.plot(np.arange(0, data.shape[-1]*dt, dt), data)
            axes[0, 0].set_title("Time domain")#, segment {}".format(i))
        
            sigma = 50
            smoothed_curve = gaussian_filter(data, sigma=sigma)
            ax = axes[i, 1]
            ax.plot(np.arange(0, data.shape[-1]*dt, dt), smoothed_curve)
            axes[0, 1].set_title("Time domain")

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
            axes[0, 2].set_title("Freq domain scaled")#, segment {}".format(i))

            ax = axes[i, 3]
            ax.plot(faxis[:50], spectrum[:50])
            axes[0, 3].set_title("Freq domain NOT scaled")#, segment {}".format(i))
    if folder[x:x+4] == "BMR2":
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

