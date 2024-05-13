from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_widths
import numpy as np
import glob
import os
import random


def get_wav_data(file_path):
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)
    # print(sample_rate)
    # print(len(data), data.dtype)
    # Calculate duration
    duration = len(data) / float(sample_rate)

    return duration, sample_rate, data


def plot_spectograph_spectogram(folders_to_iter):

    for folder in folders_to_iter:
        #print(folder)
        files = glob.glob(os.path.join(folder, '*'))

        fig, axes = plt.subplots(len(files), 5, figsize=(18,10))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)

        # Loop over each file
        for i, file in enumerate(files):

            duration, sample_rate, data = get_wav_data(file)            
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
            spec, frequencies, times, image = ax.specgram(data, NFFT=1024, Fs=sample_rate)
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
    

def find_peak_pattern(wav_data, div_width_by=1.5):

    # Find the indices of the peaks in the filtered data
    filtered_peaks = find_peaks(wav_data)[0]
    # Assuming the "best peak" is the one with the greatest amplitude, find the index of that peak
    best_peak_index = wav_data[filtered_peaks].argmax()
    # print(filtered_peaks, smoothed_curve[filtered_peaks], best_peak_index)

    # Find the estimated widths of each peak (similar to wavelength)
    peak_widths_data = peak_widths(wav_data, filtered_peaks)[0]
    # Isolating the peak width of the "best peak"
    peak_wid = peak_widths_data[best_peak_index]
    # Finding the indices of the entire peak, from its estimated start to end, according to the peak itself and its width
    start = int(filtered_peaks[best_peak_index] - peak_wid/div_width_by) # The denominator in the division here and below can be decreased/increased if we want more/less data points around that peak
    end = int(filtered_peaks[best_peak_index] + peak_wid/div_width_by)
    # The peak interval indices will be np.arange(start, end+1, 1) but to *plot it* we need to multiply it by dt
    interval = np.arange(start, end+1, 1)
    
    idealized_peak = wav_data[interval]
    
    # Find the height of the peak
    peak_height = idealized_peak.max() - np.mean(np.array([idealized_peak[0], idealized_peak[-1]]))
    return idealized_peak, interval, peak_wid, peak_height, filtered_peaks, best_peak_index


def plot_peak_patterns(folders_to_iter, sigma=30, plots=False):

    for folder in folders_to_iter:
        files = glob.glob(os.path.join(folder, '*'))
        
        if plots:
            fig, axes = plt.subplots(len(files), 2, figsize=(18,10))
            plt.subplots_adjust(wspace=0.4, hspace=0.3)

        # Loop over each file
        for i, file in enumerate(files):
            
            # Get the wav file data
            duration, sample_rate, data = get_wav_data(file)     
            dt = 1/sample_rate # difference from one time point to the next
            # Find the peaks present in our data - NON FILTERED 
            # peaks, _ = find_peaks(data) # Commented out because I only want the idealized peak of the filtered data 
            
            # Filter data (gaussian filter)
            smoothed_curve = gaussian_filter(data, sigma=sigma)
            # Get peak data
            idealized_peak, interval, peak_wid, peak_height, filtered_peaks, best_peak_index =  find_peak_pattern(smoothed_curve)

            if plots:
                # Plot the raw data
                ax = axes[i, 0]
                ax.plot(np.arange(0, data.shape[-1]*dt, dt), data)
                ax.set_ylabel('Voltage (V)')
                axes[0, 0].set_title("Time domain")
                
                # Plot the filtered data
                ax = axes[i, 1]
                ax.plot(np.arange(0, data.shape[-1]*dt, dt), smoothed_curve)
                ax.set_ylabel('Voltage (V)')
                axes[0, 1].set_title("Time domain (filtered)")
                
                # Visualizing the maximal peak
                ax.scatter(filtered_peaks[best_peak_index]*dt, smoothed_curve[filtered_peaks[best_peak_index]], c='r')
                ax.plot(interval*dt, idealized_peak, ls='--', c='crimson', lw=3)
            
        if plots: 
            fig.suptitle(folder)
            plt.show()


def get_waves_and_labels(folders_to_iter):

    n_samples = len(folders_to_iter)
    communicators = np.empty(shape=(n_samples, 2), dtype='<U4')
    labeled_wavs = [[None, None, None, None] for _ in range(n_samples)]

    # Loop over all the interaction folders
    for j, folder in enumerate(folders_to_iter):
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
            duration, sample_rate, data = get_wav_data(file)  
            wav_array[i] = data

        labeled_wavs[j][0] = wav_array
        labeled_wavs[j][1], labeled_wavs[j][2] = communicators[j, 0], communicators[j, 1]
        labeled_wavs[j][3] = parts[-1] # saving the sample number - not mandatory

    return labeled_wavs


def add_to_dict(data_dict, filtered_dict, peaks_dict, intervals_dict, widths_dict, heights_dict, key, data_list, filtered_list, peaks_list, intervals_list, widths_list, heights_list):
    if key not in data_dict:
        data_dict[key] = data_list
        filtered_dict[key] = filtered_list
        peaks_dict[key] = peaks_list
        intervals_dict[key] = intervals_list
        widths_dict[key] = widths_list
        heights_dict[key] = heights_list
    else:
        data_dict[key].extend(data_list)
        filtered_dict[key].extend(filtered_list)
        peaks_dict[key].extend(peaks_list)
        intervals_dict[key].extend(intervals_list)
        widths_dict[key].extend(widths_list)
        heights_dict[key].extend(heights_list)


def combine_wavs_by_communicators(folders_to_iter, sigma=30):
    '''Groups all arrays of wav data in dictionaries by:
        1. noise makers
        2. noise receivers
        3. both labels
        Also, groups all arrays of the peaks in dictionaries in the same order.
        Finally returns a dictionary: 
        {
            'by_maker' : {'grouped_data' : noise makers data dictionary, ..., # SORRY IM TOO LAZY TO TYPE ALL JUST PRINT IT OUT POR FAVORRRRR :)
                            'grouped_peaks' : noise makers peaks dictionary},
            'by_receiver' : {'grouped_data' : noise receivers data dictionary, ...,
                            'grouped_peaks' : noise receivers peaks dictionary},
            'by_both' : {'grouped_data' : both labels data dictionary, ...,
                        'grouped_peaks' : both labels peaks dictionary}
        }
        '''
    
    labeled_wavs = get_waves_and_labels(folders_to_iter)
    data_types = ['grouped_data', 'filtered_data', 'grouped_peaks', 'grouped_intervals', 'grouped_widths', 'grouped_heights']
    group_by = ['by_maker', 'by_receiver', 'by_both']

    final = {group : {data_type : {} for data_type in data_types} for group in group_by}

    for wav_data, noise_maker, noise_receiver, _ in labeled_wavs:
        peaks = []
        intervals = []
        peak_widths = []
        peak_heights = []
        filtered = []
        for wav_array in wav_data:
            filtered_data = gaussian_filter(wav_array, sigma=sigma)
            filtered.append(filtered_data)
            peak, interval, peak_wid, peak_height = find_peak_pattern(filtered_data)[:4]
            peaks.append(peak)
            intervals.append(interval)
            peak_widths.append(peak_wid)
            peak_heights.append(peak_height)

        all_data = [wav_data, filtered, peaks, intervals, peak_widths, peak_heights]
        # Group WAV data arrays by noise maker and receiver names
        keys = [noise_maker, noise_receiver, (noise_maker, noise_receiver)]
        for i, group in enumerate(group_by):
            D = final[group]
            add_to_dict(*[D[data_type] for data_type in data_types], keys[i], *all_data)

        # for d in wav_data: # This nested loop proves the need for using lists instead of normal organized arrays :(((
        #     if len(d) != 2500:
        #         print(noise_maker, noise_receiver, len(d))

    return final 


def plot_peak_on_grouped_data(final_dict, sample_rate=44100, by='by_both', n_rows=5):
    
    data_dict = final_dict[by]
    keys = data_dict['grouped_data'].keys()
    dt = 1/sample_rate

    for i, individual in enumerate(keys):
        n_samples = min(n_rows, len(data_dict['grouped_data'][individual]))
        fig, axes = plt.subplots(n_samples, 2, figsize=(12,8))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        raw_data = data_dict['grouped_data'][individual]
        filtered_data = data_dict['filtered_data'][individual]
        peaks = data_dict['grouped_peaks'][individual]
        peak_intervals = data_dict['grouped_intervals'][individual]
        peak_heights = data_dict['grouped_heights'][individual]

        axes[0,0].set_title('Unfiltered Signal')
        axes[0,1].set_title('Filtered Signal')
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        axes[n_samples//2, 0].set_ylabel('Amplitude (Voltage)')

        for sample in range(n_samples):

            ax = axes[sample, 0]
            ax.plot(np.arange(0, raw_data[sample].shape[-1])*dt, raw_data[sample])
            ax.plot(peak_intervals[sample]*dt, peaks[sample], c='r', lw=4, alpha=0.6)
            
            ax = axes[sample, 1]
            ax.plot(np.arange(0, raw_data[sample].shape[-1])*dt, filtered_data[sample])
            ax.plot(peak_intervals[sample]*dt, peaks[sample], c='r', lw=4, alpha=0.6)
            
            print(peak_heights[sample])

        fig.suptitle(individual)
        ax.legend(['data', 'idealized peak'], loc='best')
        plt.show()


random.seed(5)

folder_path = '/Users/Omer/Documents/DirectPhD/BSA/Final project/Signals/Segmented_signals'

# Get a list of all files in the folder matching a specific pattern
folders = glob.glob(os.path.join(folder_path, '*'))
folders_to_iter = random.sample(folders, 8)

# plot_spectograph_spectogram(folders_to_iter)
# plot_peak_patterns(folders_to_iter, plots=True)
# print(get_waves_and_labels(folders_to_iter))
# print(combine_wavs_by_communicators(folders_to_iter)['by_maker']['grouped_peaks'].keys())
final_dict = combine_wavs_by_communicators(folders_to_iter)
# print(len(final_dict['by_maker']['grouped_peaks']['BMR2']))
plot_peak_on_grouped_data(final_dict)