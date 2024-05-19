import os
import glob
import random
import platform
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_widths
from scipy.signal import convolve

def main():
    random.seed(5)
    
    if platform.system() == 'Windows':
        folder_path = os.path.normcase('C:/Data/BSA_proj/Segmented_signals')
    else:
        folder_path = os.path.normcase('/Users/Omer/Downloads/Segmented_signals/')

    # Get a list of all files in the folder matching a specific pattern
    folders = glob.glob(os.path.join(folder_path, '*'))
    folders_to_iter = random.sample(folders, 20)

    # plot_spectograph_spectogram(folders_to_iter)
    # plot_peak_patterns(folders_to_iter, plots=True)
    # print(get_waves_and_labels(folders_to_iter))
    # print(combine_wavs_by_communicators(folders_to_iter)['by_maker']['grouped_peaks'].keys())


    final_dict = combine_wavs_by_communicators(folders_to_iter)

    ideal_picks = get_individual_ideal_peak(final_dict, peak_width_determination_method='mean')
    #ideal_picks = get_individual_ideal_peak(final_dict, peak_width_determination_method='min')

    # print(len(final_dict['by_maker']['grouped_peaks']['BMR2']))
    # plot_peak_on_grouped_data(final_dict)


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
    # print('peaks:', filtered_peaks)#, '')
    # Assuming the "best peak" is the one with the greatest amplitude, find the index of that peak
    best_peak_index = wav_data[filtered_peaks].argmax()
    # print(filtered_peaks, smoothed_curve[filtered_peaks], best_peak_index)

    # Find the estimated widths of each peak (similar to wavelength)
    peak_widths_data = peak_widths(wav_data, filtered_peaks)[0]
    # Isolating the peak width of the "best peak"
    peak_wid = peak_widths_data[best_peak_index]
    # Finding the indices of the entire peak, from its estimated start to end, according to the peak itself and its width
    start = int(filtered_peaks[best_peak_index] - peak_wid/div_width_by) # The denominator in the division here and below can be decreased/increased if we want more/less data points around that peak
    end = min(int(filtered_peaks[best_peak_index] + peak_wid/div_width_by), len(wav_data))
    # The peak interval indices will be np.arange(start, end+1, 1) but to *plot it* we need to multiply it by dt
    interval = np.arange(start, end, 1)
    # print(interval)
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
    total_len = 0

    # Loop over all the interaction folders
    for j, folder in enumerate(folders_to_iter):
        if type(folder) != str:    
            print(folder)
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
            total_len += 1

        labeled_wavs[j][0] = wav_array
        labeled_wavs[j][1], labeled_wavs[j][2] = communicators[j, 0], communicators[j, 1]
        labeled_wavs[j][3] = files # saving the file name - not mandatory
    
    # print(total_len)

    return labeled_wavs, total_len

def get_waves_and_labels_from_files(files):

    n_files = len(files)
    communicators = np.empty(shape=(n_files, 2), dtype='<U4')
    labeled_wavs = [[None, None, None, None] for _ in range(n_files)]

    # Loop over all the interaction folders
    for i, file in enumerate(files):
        # Split folder path using underscores
        parts = os.path.basename(file[:file.rfind('/')]).split('_')
        # print(parts)
        
        # Extract individuals' names
        noise_maker = parts[0]
        noise_receiver = parts[2]
    
        # Save the individuals communicating
        communicators[i, 0] = noise_maker
        communicators[i, 1] = noise_receiver
        
        # Get the wav data for each segmented file in each folder 
        duration, sample_rate, data = get_wav_data(file)  
        wav_array = [data]
        labeled_wavs[i][0] = wav_array
        labeled_wavs[i][1], labeled_wavs[i][2] = communicators[i, 0], communicators[i, 1]
        labeled_wavs[i][3] = [file] # saving the file name - not mandatory

    return labeled_wavs

def add_to_dict(dicts_list, key, list_of_data_lists): #data_dict, filtered_dict, peaks_dict, intervals_dict, widths_dict, heights_dict, key, data_list, filtered_list, peaks_list, intervals_list, widths_list, heights_list):
    ''' dicts_list = [data_dict, filtered_dict, peaks_dict, intervals_dict, widths_dict, heights_dict, sample_numbers_dict]
        lists_for_appending = [data_list, filtered_list, peaks_list, intervals_list, widths_list, heights_list, sample_numbers_list] '''
    for i, D in enumerate(dicts_list):    
        data_list = list_of_data_lists[i]
        if key not in D.keys():
            D[key] = data_list
        else:
            D[key].extend(data_list)
        

def combine_wavs_by_communicators(to_iter, sigma=20, file_list=False):
    '''Groups all arrays of wav data in dictionaries by:
        1. noise makers
        2. noise receivers
        3. both labels
        Also, groups all arrays of the peaks in dictionaries in the same order.
        Finally returns a dictionary: 
        {
            'by_maker' : 
            {
                'grouped_data' : the data in the wav file, all the files from all folders grouped by noise maker,
                'filtered_data' : the same as in grouped_data but with gaussian filter applied,
                'grouped_peaks' : list of the peak values from the file,
                'grouped_intervals' : list of the peak indexes as they appear in the filtered data and the grouped data lists
                'grouped_widths' : list of the peak widths (how long in time the peak lasted),
                'grouped_heights' : list of the peak heights (the amplitude of the peak as the distance from the noise floor), 
                'sample_numbers' : 
            },
            'by_receiver' : { The same as above but grouped by noise receiver},
            'by_both' : { The same as above but grouped by both noise maker and receiver}
        }
    '''
    if file_list:
        labeled_wavs = get_waves_and_labels_from_files(to_iter)
    else: 
        labeled_wavs = get_waves_and_labels(to_iter)[0]
    data_types = ['grouped_data', 'filtered_data', 'grouped_peaks', 'grouped_intervals', 'grouped_widths', 'grouped_heights', 'file_names']
    group_by = ['by_maker', 'by_receiver', 'by_both']

    final = {group : {data_type : {} for data_type in data_types} for group in group_by}

    for wav_data, noise_maker, noise_receiver, files in labeled_wavs:
        peaks = []
        intervals = []
        peak_widths = []
        peak_heights = []
        filtered = []
        file_names = []
        for i, wav_array in enumerate(wav_data):
            filtered_data = gaussian_filter(wav_array, sigma=sigma)
            filtered.append(filtered_data)
            peak, interval, peak_wid, peak_height = find_peak_pattern(filtered_data)[:4]
            peaks.append(peak)
            intervals.append(interval)
            peak_widths.append(peak_wid)
            peak_heights.append(peak_height)
            file_names.append(files[i])

        all_data = [wav_data, filtered, peaks, intervals, peak_widths, peak_heights, file_names]
        # Group WAV data arrays by noise maker and receiver names
        keys = [noise_maker, noise_receiver, (noise_maker, noise_receiver)]
        for i, group in enumerate(group_by):
            D = final[group]
            add_to_dict([D[data_type] for data_type in data_types], keys[i], all_data)

        # for d in wav_data: # This nested loop proves the need for using lists instead of normal organized arrays :(((
        #     if len(d) != 2500:
        #         print(noise_maker, noise_receiver, len(d))

    return final 


def get_individual_ideal_peak(combined_wavs_by_communicators, peak_width_determination_method='min'):
    '''
    Description
    ------------
    Based on the data from the combined_wavs_by_communicators function, this function will calculate the ideal peak width for each individual,
    to be then used to cross correlate against to establish the probability distribution of whether a recording is from the same individual or not.
    Same logic goes for whom was addressed in the recording.

    Parameters
    ------------
    combined_wavs_by_communicators : dict
        The dictionary returned by the combined_wavs_by_communicators function
    method : str
        The method used to calculate the ideal peak width, either 'mean', 'median' or 'min'
        mean will calculate the mean width of peaks, then pad or trim peaks symmetrically to this width,
        median will calculate the median width of peaks, then pad or trim peaks simmetrically to this width,
        min will calculate the minimum width peak and trim all other peaks to this width.
        Irespective of the method, the cleaned list of peaks will be the same length for an individual and will then be averaged to get the ideal peak.

    Returns
    ------------
    ideal_peaks_noise_maker : dict
        A dictionary with the noise maker as the key and the ideal peak as the value
    ideal_peaks_noise_receiver : dict
        A dictionary with the noise receiver as the key and the ideal peak as the value
    '''
    # Prepare the data for the function calls to find the ideal peaks
    # For the noise maker
    noise_maker_filtered_data_dict  = combined_wavs_by_communicators['by_maker']['filtered_data']
    noise_maker_peaks_indexes = combined_wavs_by_communicators['by_maker']['grouped_intervals']
    ideal_peaks_noise_maker = __find_ideal_peaks(noise_maker_filtered_data_dict, noise_maker_peaks_indexes, peak_width_determination_method)

    # For the noise receiver
    noise_receiver_filtered_data_dict = combined_wavs_by_communicators['by_receiver']['filtered_data']
    noise_receiver_peaks_indexes = combined_wavs_by_communicators['by_receiver']['grouped_intervals']
    ideal_peaks_noise_receiver = __find_ideal_peaks(noise_receiver_filtered_data_dict, noise_receiver_peaks_indexes, peak_width_determination_method)

    return ideal_peaks_noise_maker, ideal_peaks_noise_receiver


def __find_ideal_peaks(filtered_data, grouped_intervals, peak_width_determination_method):
    ideal_peaks = {}
    
    for noise_maker_key in filtered_data.keys():
        
        # Grab the current mole rat data from all preprepared dictionaries
        current_individual_filtered_data = filtered_data[noise_maker_key]
        current_individual_peaks_indexes = grouped_intervals[noise_maker_key]

        # Set a variable to use the chosen method to later call with the array of peak widths
        if peak_width_determination_method == 'min':
            peak_width_determination_method_func = np.min
        elif peak_width_determination_method == 'mean':
            peak_width_determination_method_func = np.mean
        elif peak_width_determination_method == 'median':
            peak_width_determination_method_func = np.median
        else:
            raise NotImplementedError(f'Method {peak_width_determination_method} is not implemented. See documetation for supported methods.')

        # Find the peak width with the chosen method
        current_ideal_peak_width = int(peak_width_determination_method_func([len(peak) for peak in current_individual_peaks_indexes]))
        # The amount of padding from the left and right of the peak to make it the ideal peak width
        pad_width = current_ideal_peak_width // 2


        # Trim all peaks to the determined width around the index of the peak
        trimmed_peaks = []
        for peak_data, curr_inteval in zip(current_individual_filtered_data, current_individual_peaks_indexes):

            # Find the middle value of the inrevals (the indexes in the filtered data where the peaks are located)
            mid_peak_index = len(curr_inteval)//2

            # Save the indexes so they can be changed in the case of an ideal peak width that required overflow
            # This will happen in cases where pad_width >= len(indexes)//2
            trim_strat_interval_index = mid_peak_index - pad_width
            trim_end_interval_index = mid_peak_index + pad_width

            # Check if the slices are within the bounds of the data
            # strat zeros in case of overflow to the left
            strat_zeros = []
            
            if trim_strat_interval_index < 0:
                strat_zeros = np.zeros(np.abs(trim_strat_interval_index))
                # Set the start point for taking values from the peak at the start of the peak
                trim_strat_interval_index = 0

            # Same logic for the end zeros
            end_zeros = []

            if trim_end_interval_index > len(curr_inteval) - 1:
                # +1 since the otherwise the end_zeros will be one less than the required
                end_zeros = np.zeros(trim_end_interval_index - len(curr_inteval) + 1)
                # Set the end point for taking values from the peak at the end of the peak
                trim_end_interval_index = len(curr_inteval) - 1
            
            
            # print('len:', len(curr_inteval), '. start:', trim_strat_interval_index, '. end:', trim_end_interval_index)
            # Add the zero arrays to the current trimmed peak
            trimmed_peak = np.concatenate([strat_zeros, peak_data[curr_inteval[trim_strat_interval_index]:curr_inteval[trim_end_interval_index]], end_zeros])
            trimmed_peaks.append(trimmed_peak)


        # Average the trimmed peaks
        ideal_peak = np.mean(trimmed_peaks, axis=0)
        # Save the ideal peak
        ideal_peaks[noise_maker_key] = ideal_peak


    return ideal_peaks


def convolve_ideal_peak_over_filtered_data(final_dict, ideal_peaks, by='by_maker'):
    
    data_dict = final_dict[by]
    data_keys = data_dict['grouped_data'].keys()
    print(data_keys)
    peak_keys = ideal_peaks.keys()
    convolution_results = {peak_key : {} for peak_key in peak_keys}

    for peak_individual in peak_keys:
        peak = ideal_peaks[peak_individual]
        curr_dict = convolution_results[peak_individual]
        
        for data_individual in data_keys:
            # print(data_individual)
            filtered_data_peaks = data_dict['grouped_peaks'][data_individual]

            for i, sample in enumerate(filtered_data_peaks):
                matched_filter_output = convolve(sample, peak, mode='same')
                # print(list(matched_filter_output))

                if data_individual not in curr_dict.keys():
                    curr_dict[data_individual] = [np.max(matched_filter_output)]
                else:
                    curr_dict[data_individual].append(np.max(matched_filter_output))
                # plt.plot(np.arange(matched_filter_output.shape[-1]), matched_filter_output)
                # plt.show()
    
    # total_scores = 
    print(convolution_results.keys())
    fig, axes = plt.subplots(len(peak_keys), 1, sharex=True)
    plt.subplots_adjust(hspace=0.3)
    conv_keys = peak_keys
    colors = ['red', 'forestgreen', 'blue', 'black']
    cnt = 0
    # conv_res = list(convolution_results.values())
    # print(conv_res[0])
    for i, peak_ind in enumerate(conv_keys):
        ax = axes[i]
        # for tested_pair, convolution_scores in convolution_results.items():
        # conv_res = convolution_results[conv_keys[i]]
        for data_ind, conv_res in convolution_results[peak_ind].items():
            ax.hist(conv_res, bins=35, color=colors[cnt], label=data_ind, alpha=0.5)
            ax.set_title(f'Peak individual: {peak_ind}')# ; Actual data is of {data_ind}')
            cnt +=1
        cnt=0
    ax.legend()
    ax.set_xlabel('Convolution Score')
    plt.show()
        
        


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
            
            # print(peak_heights[sample])

        fig.suptitle(individual)
        ax.legend(['data', 'idealized peak'], loc='best')
        plt.show()


def convolution_results_hist_vals(grouped_peaks_dict, ideal_peaks):

    data_keys = grouped_peaks_dict.keys()
    # print(data_keys)
    peak_keys = ideal_peaks.keys()
    convolution_results = {peak_key : {} for peak_key in peak_keys}

    for data_ind, data_peaks in grouped_peaks_dict.items():
        
        for peak_ind, peak in ideal_peaks.items():
            curr_dict = convolution_results[peak_ind]
            
            for i, sample in enumerate(data_peaks):
                # print(sample, peak)
                matched_filter_output = convolve(sample, peak, mode='same')
                # print(list(matched_filter_output))

                if data_ind not in curr_dict.keys():
                    curr_dict[data_ind] = [np.max(matched_filter_output)]
                else:
                    curr_dict[data_ind].append(np.max(matched_filter_output))
    
    return convolution_results


def make_distribution(train, test, by='by_maker', peak_width_determination_method='mean', file_list=False, sigma=20):

    train_dict = combine_wavs_by_communicators(train, sigma=sigma, file_list=file_list)
    test_dict = combine_wavs_by_communicators(test, sigma=sigma, file_list=file_list)
    
    ideal_peaks = __find_ideal_peaks(train_dict[by]['filtered_data'], train_dict[by]['grouped_intervals'], peak_width_determination_method)
    test_peaks_dict = test_dict[by]['grouped_peaks']

    train_convolution_results = convolution_results_hist_vals(train_dict[by]['grouped_peaks'], ideal_peaks)
    test_convolution_results = convolution_results_hist_vals(test_peaks_dict, ideal_peaks)
    # print(train_convolution_results)
    # for i, ind1 in enumerate(train_convolution_results.keys()):
    #     for j, ind2 in enumerate(ideal_peaks.keys()):
    #         #print(train_convolution_results)
    #         plt.hist(train_convolution_results[ind1][ind2], label='train', alpha=0.5)
    #         plt.hist(test_convolution_results[ind1][ind2], color='r', label='test', alpha=0.5)
    #         plt.xlabel('Conv score')
    #         plt.title(f'tested individual : {ind1} ; ideal peak by : {ind2}')
    #         plt.legend()

    #         plt.show()
    
    return train_convolution_results, test_convolution_results

def split_test_train_folders(folders_to_iter, n_tests):
    
    n_folders = len(folders_to_iter)
    all_file_data, n_files = get_waves_and_labels(folders_to_iter)
    labels_by_makers = []
    labels_by_receivers = []
    
    for wav_data, noise_maker, noise_receiver, sample_number in all_file_data:
        if noise_maker not in labels_by_makers:
            labels_by_makers.append(noise_maker)
        if noise_receiver not in labels_by_receivers:
            labels_by_receivers.append(noise_receiver)
    
    n_files_per_ind_by_maker = n_tests // len(labels_by_makers)
    n_files_per_ind_by_receiver = n_tests // len(labels_by_receivers)

    n_files_by_makers = [n_files_per_ind_by_maker if i != len(labels_by_makers)-1 else (n_tests - n_files_per_ind_by_maker * (len(labels_by_makers) - 1)) for i in range(len(labels_by_makers))]
    n_files_by_receiver = [n_files_per_ind_by_receiver if i != len(labels_by_receivers)-1 else (n_tests - n_files_per_ind_by_receiver * (len(labels_by_receivers) - 1)) for i in range(len(labels_by_receivers))]

    final_dict = combine_wavs_by_communicators(folders_to_iter)
    dict_by_maker_filename = final_dict['by_maker']['file_names']
    dict_by_receiver_filename = final_dict['by_receiver']['file_names']

    test_data_by_makers = [] #{maker : dict_by_maker_filename[maker][:n] for n in n_files_by_makers for maker in labels_by_makers}
    test_data_by_receivers = [] #{receiver : dict_by_receiver_filename[receiver][:n] for n in n_files_by_receiver for receiver in labels_by_receivers}

    train_data_by_makers = [] #{maker : dict_by_maker_filename[maker][n:] for n in n_files_by_makers for maker in labels_by_makers}
    train_data_by_receivers = [] #{receiver : dict_by_receiver_filename[receiver][n:] for n in n_files_by_receiver for receiver in labels_by_receivers}

    for i, maker in enumerate(labels_by_makers):
        n = n_files_by_makers[i]
        test_data_by_makers.extend(dict_by_maker_filename[maker][:n])
        train_data_by_makers.extend(dict_by_maker_filename[maker][n:])
        # print(len(dict_by_maker_filename[maker]))

    for i, receiver in enumerate(labels_by_receivers):
        n = n_files_by_receiver[i]
        test_data_by_receivers.extend(dict_by_receiver_filename[receiver][:n])
        train_data_by_receivers.extend(dict_by_receiver_filename[receiver][n:])
        # print(len(dict_by_receiver_filename[receiver]))

    # print(n_files)
    
    return test_data_by_makers, train_data_by_makers, test_data_by_receivers, train_data_by_receivers


def percentiles(train_convolution_results, test_convolution_results, ideal_peaks, percentile_threshold=90):
    
    tested_inds_accuracies = {tested_ind : {peak_ind : 0 for peak_ind in ideal_peaks.keys()} for tested_ind in test_convolution_results.keys()}
    for train_ind, convolution_scores_dict in train_convolution_results.items():
        for peak_ind in ideal_peaks.keys():
            value_threshold = np.percentile(np.array(train_convolution_results[train_ind][peak_ind]), percentile_threshold)
            test_ind_array = np.array(test_convolution_results[train_ind][peak_ind])
            test_bigger_than_threshold = np.mean(test_ind_array > value_threshold)
            # print(sum(test_ind_array > value_threshold) / len(test_ind_array), test_bigger_than_threshold)
            tested_inds_accuracies[train_ind][peak_ind] = test_bigger_than_threshold
    
    return tested_inds_accuracies


random.seed(7)

folder_path = '/Users/Omer/Documents/DirectPhD/BSA/Final project/Signals/Segmented_signals'

# Get a list of all files in the folder matching a specific pattern
folders = glob.glob(os.path.join(folder_path, '*'))
folders_to_iter = folders # random.sample(folders, 200)
# labeled_wavs = get_waves_and_labels(folders_to_iter)
final_dict = combine_wavs_by_communicators(folders_to_iter)
# print(final_dict['by_receiver']['grouped_data'].keys())
# print(final_dict['by_maker']['sample_numbers'])

ideal_peaks = __find_ideal_peaks(final_dict['by_maker']['filtered_data'], final_dict['by_maker']['grouped_intervals'], 'mean')
# convolve_ideal_peak_over_filtered_data(final_dict, res, by='by_maker')

test_data_by_makers, train_data_by_makers, test_data_by_receivers, train_data_by_receivers = split_test_train_folders(folders_to_iter, 400)
# for maker, test_file_list in test_data_by_makers.items():
    # train_file_list = train_data_by_makers[maker]
train_convolution_results, test_convolution_results = make_distribution(train_data_by_makers, test_data_by_makers, file_list=True)
res = percentiles(train_convolution_results, test_convolution_results, ideal_peaks, percentile_threshold=80)
print(res)
fig, axes = plt.subplots(len(res), 1)
plt.subplots_adjust(hspace=0.5)
for i, tested_ind in enumerate(res.keys()):
    ax = axes[i]
    x_labels = res[tested_ind].keys()
    heights = res[tested_ind].values()
    ax.bar(x_labels, heights, color='r', alpha=0.5)
    ax.set_title(f'tested individual is {tested_ind}')
    # ax.set_xticklabels(x_labels)
    # ax.set_ylabel('')
plt.show()



# if __name__ == '__main__':
#     main()