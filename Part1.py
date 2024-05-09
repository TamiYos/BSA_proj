from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import glob
import os

def get_wav_duration(file_path):
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)
    # print(sample_rate)
    # print(len(data), data.dtype)

    # Calculate duration
    duration = len(data) / float(sample_rate)

    return duration, sample_rate, data

def get_waves_and_labels(folder_path):
    # Get all the folders inside the main folder
    folders = glob.glob(os.path.join(folder_path, '*'))
    n_samples = len(folders)
    communicators = np.empty(shape=(n_samples, 2), dtype='<U4')
    labeled_wavs = [[None, None, None] for _ in range(n_samples)]

    # Loop over all the interaction folders
    for j, folder in enumerate(folders):
        # Split folder path using underscores
        parts = os.path.basename(folder).split('_')
        
        # Extract individuals' names
        noise_maker = parts[0]
        noise_receiver = parts[2]
    
        # Save the individuals communicating
        communicators[j, 0] = noise_maker
        communicators[j, 1] = noise_receiver

        # Retrieve the communication files in that folder ('segmented_i.wav')
        files = glob.glob(os.path.join(folder, '*'))
        n_files = len(files)
        
        # Get the wav data for each segmented file in each folder 
        wav_array = [None for _ in range(n_files)]
        for i, file in enumerate(files):
            duration, sample_rate, data = get_wav_duration(file)  
            wav_array[i] = data
        
        labeled_wavs[j][0] = wav_array
        labeled_wavs[j][1], labeled_wavs[j][2] = communicators[j, 0], communicators[j, 1]

    return labeled_wavs

def combine_wavs_by_communicators(folder_path):
    
    labeled_wavs = get_waves_and_labels(folder_path)
    combined_wavs = {}
    
    # Group WAV data arrays by noise maker and receiver names
    for wav_data, noise_maker, noise_receiver in labeled_wavs:
        # for d in wav_data: # This nested loop proves the need for using lists instead of normal organized arrays :(((
        #     if len(d) != 2500:
        #         print(noise_maker, noise_receiver, len(d))
        key = (noise_maker, noise_receiver)
        if key not in combined_wavs:
            combined_wavs[key] = wav_data
        else:
            combined_wavs[key].extend(wav_data)

    # Convert the dictionary to the desired format
    final_combined_wavs = []
    for key, wav_arrays in combined_wavs.items():
        noise_maker, noise_receiver = key
        final_combined_wavs.append([wav_arrays, noise_maker, noise_receiver])
    
    return final_combined_wavs

def combine_wavs_by_makers(folder_path):

    labeled_wavs = get_waves_and_labels(folder_path)
    combined_wavs = {}
    
    # Group WAV data arrays by noise maker and receiver names
    for wav_data, noise_maker, noise_receiver in labeled_wavs:
        key = noise_maker
        if key not in combined_wavs:
            combined_wavs[key] = wav_data
        else:
            combined_wavs[key].extend(wav_data)

    # Convert the dictionary to the desired format
    final_combined_wavs = []
    for key, wav_arrays in combined_wavs.items():
        noise_maker = key
        final_combined_wavs.append([wav_arrays, noise_maker])
    
    # print(combined_wavs.keys())
    return final_combined_wavs


folder_path = '/Users/Omer/Documents/DirectPhD/BSA/Final project/Signals/Segmented_signals/'
# print(get_waves_and_labels(folder_path)[-1])
# get_waves_and_labels(folder_path)
print(combine_wavs_by_communicators(folder_path))
# print([[len(combine_wavs_by_communicators(folder_path)[j][0][i]) for i in range(9)] for j in range(len(combine_wavs_by_communicators(folder_path)))])
# combine_wavs_by_makers(folder_path)
