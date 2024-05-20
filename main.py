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

    # TODO, change this before running the code to the folder path of Segmented_signals (from the Mole Rat project in the drive)
    # This is the only needed change (:
    folder_path = 'C:/Data/BSA_proj/Segmented_signals'
    
    # Get a list of all files in the folder matching a specific pattern
    folders = glob.glob(os.path.join(folder_path, '*'))
    folders_to_analyse = [folder for folder in folders if 'json' not in folder and 'csv' not in folder]

    # Get the data from the files into the needed data structure for ease of use
    final_dict = combine_wavs_by_communicators(folders_to_analyse)
    
    # Get the idealized peaks for each individual under each role
    ideal_peaks_maker, ideal_peaks_receiver = get_individual_ideal_peak(final_dict, peak_width_determination_method='mean')

    # Split the data into test and train sets
    test_data_by_makers, train_data_by_makers, test_data_by_receivers, train_data_by_receivers = split_test_train_folders(folders_to_analyse, 400)

    # Generate the probability distribution of the convolution scores to later on make predictions
    train_convolution_results_maker, test_convolution_results_maker = make_distribution(train_data_by_makers, test_data_by_makers, file_list=True)
    
    # See where the test scores fall in the distribution of the train scores and if they are above a certain percentile threshold
    res_maker = percentiles(train_convolution_results_maker, test_convolution_results_maker, ideal_peaks_maker, percentile_threshold=85)

    # Do the same as above but for the noise receivers
    train_convolution_results_receiver, test_convolution_results_receiver = make_distribution(test_data_by_receivers, train_data_by_receivers, by='by_receiver', file_list=True)
    res_recipient = percentiles(train_convolution_results_receiver, test_convolution_results_receiver, ideal_peaks_receiver, percentile_threshold=85)
    
    # Plot the results
    fig, axes = plt.subplots(len(res_maker), 1, figsize=(10,15))
    plt.subplots_adjust(hspace=0.5)
    colors = ['indianred', 'darkorange', 'teal', 'royalblue']
    for i, tested_ind in enumerate(res_maker.keys()):
        ax = axes[i]
        x_labels = res_maker[tested_ind].keys()
        heights = res_maker[tested_ind].values()
        ax.bar(x_labels, heights, color=colors[i], alpha=0.5)
        ax.set_title(f'Tested noise maker is {tested_ind}', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
    axes[-1].set_xlabel('Ideal peak individual', fontsize=14)
    axes[len(res_maker)//2].set_ylabel('Fraction over percentile threshold', fontsize=14)
    fig.suptitle('Fraction of convolution scores per noise maker\nover percentile threshold of 85%', fontsize=18)
    plt.show()

    fig, axes = plt.subplots(len(res_recipient), 1, figsize=(10,15))
    plt.subplots_adjust(hspace=0.5)
    colors = ['indianred', 'darkorange', 'teal', 'royalblue', 'indigo']
    for i, tested_ind in enumerate(res_recipient.keys()):
        ax = axes[i]
        x_labels = res_recipient[tested_ind].keys()
        heights = res_recipient[tested_ind].values()
        ax.bar(x_labels, heights, color=colors[i], alpha=0.5)
        ax.set_title(f'Tested noise listener is {tested_ind}', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
    axes[-1].set_xlabel('Ideal peak individual', fontsize=14)
    axes[len(res_recipient)//2].set_ylabel('Fraction over percentile threshold', fontsize=14)
    fig.suptitle('Fraction of convolution scores per noise listener\nover percentile threshold of 85%', fontsize=18)
    plt.show()


def get_wav_data(file_path):
    '''
    Description
    ------------
    This function reads a WAV file and returns the duration, sample rate and the data in the file.

    Parameters
    ------------
    file_path : str
        The path to the WAV file to be read.
    
    Returns
    ------------
    duration : float
        The duration of the WAV file in seconds.
    sample_rate : int
        The sample rate of the WAV file.
    data : np.array
        The data in the WAV file.
    '''
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)
    # Calculate duration
    duration = len(data) / float(sample_rate)

    return duration, sample_rate, data


def find_peak_pattern(wav_data, div_width_by=1.5):
    '''
    Description
    ------------
    Get the most prominent peak in the data (as the one with the greatest amplitude) and return it along with the interval it is in,
    the width of the peak and the height of the peak.

    Parameters
    ------------
    wav_data : np.array
        The data in which to find the peak pattern.
    div_width_by : float
        The value to divide the peak width by to get the interval around the peak.
    
    Returns
    ------------
    idealized_peak : np.array
        The peak data.
    interval : np.array
        The interval in which the peak is located.
    peak_wid : float
        The width of the peak.
    peak_height : float
        The height of the peak.
    filtered_peaks : np.array
        The indices of the peaks in the filtered data.
    best_peak_index : int
        The index of the best peak.
    '''
    # Find the indices of the peaks in the filtered data
    filtered_peaks = find_peaks(wav_data)[0]
    # Assuming the "best peak" is the one with the greatest amplitude, find the index of that peak
    best_peak_index = wav_data[filtered_peaks].argmax()

    # Find the estimated widths of each peak (similar to wavelength)
    peak_widths_data = peak_widths(wav_data, filtered_peaks)[0]
    # Isolating the peak width of the "best peak"
    peak_wid = peak_widths_data[best_peak_index]
    # Finding the indices of the entire peak, from its estimated start to end, according to the peak itself and its width
    start = int(filtered_peaks[best_peak_index] - peak_wid/div_width_by) # The denominator in the division here and below can be decreased/increased if we want more/less data points around that peak
    end = min(int(filtered_peaks[best_peak_index] + peak_wid/div_width_by), len(wav_data))
    # The peak interval indices will be np.arange(start, end, 1) but to *plot it* we need to multiply it by dt
    interval = np.arange(start, end, 1)
    idealized_peak = wav_data[interval]
    
    # Find the height of the peak
    peak_height = idealized_peak.max() - np.mean(np.array([idealized_peak[0], idealized_peak[-1]]))
    return idealized_peak, interval, peak_wid, peak_height, filtered_peaks, best_peak_index


def get_waves_and_labels(folders_to_iter):
    '''
    Description
    ------------
    This function reads the WAV files in the folders and returns the data in the files along with the labels of the individuals communicating.

    Parameters
    ------------
    folders_to_iter : list
        A list of the paths to the folders containing the WAV files.
    
    Returns
    ------------
    labeled_wavs : list
        A list of the data in the WAV files along with the labels of the individuals communicating.
    '''

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
    
    return labeled_wavs, total_len


def get_waves_and_labels_from_files(files):
    '''
    Description
    ------------
    This function reads the WAV files in the given files and returns the data in the files along with the labels of the individuals communicating.

    Parameters
    ------------
    files : list
        A list of the paths to the WAV files.
    
    Returns
    ------------
    labeled_wavs : list
        A list of the data in the WAV files along with the labels of the individuals communicating.
    '''
    n_files = len(files)
    communicators = np.empty(shape=(n_files, 2), dtype='<U4')
    labeled_wavs = [[None, None, None, None] for _ in range(n_files)]

    # Loop over all the interaction folders
    for i, file in enumerate(files):

        # There what seems to be a bug in windows where the file path contains \\ instead of /, this is a workaround
        if platform.system() == 'Windows':
            file = file.replace('\\', '/')

        # Split folder path using underscores
        parts = os.path.basename(file[:file.rfind('/')]).split('_')
        
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


def combine_wavs_by_communicators(to_iter, sigma=20, file_list=False):
    '''
    Description
    ------------
    Groups all arrays of wav data in dictionaries by:
        1. noise makers
        2. noise receivers
        3. both labels

    Parameters
    ------------
    to_iter : list
        A list of the paths to the folders containing the WAV files.
    sigma : int
        The sigma value for the gaussian filter.
    file_list : bool
        A flag to indicate whether the data is in a list of files or in folders.
    
    Returns
    ------------
    dictionary with the following structure: 
    {
        'by_maker' : 
        {
            'grouped_data' : the data in the wav file, all the files from all folders grouped by noise maker,
            'filtered_data' : the same as in grouped_data but with gaussian filter applied,
            'grouped_peaks' : list of the peak values from the file,
            'grouped_intervals' : list of the peak indexes as they appear in the filtered data and the grouped data lists
            'grouped_widths' : list of the peak widths (how long in time the peak lasted),
            'grouped_heights' : list of the peak heights (the amplitude of the peak as the distance from the noise floor), 
            'file_names' : list of the file names
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
    cnt = 0

    for wav_data, noise_maker, noise_receiver, files in labeled_wavs:
        peaks = []
        intervals = []
        peak_widths = []
        peak_heights = []
        filtered = []
        file_names = []
        for i, wav_array in enumerate(wav_data):
            cnt +=1
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
            role_key = keys[i]
            dict_grouped_by_role = final[group]
            for j, data_type in enumerate(data_types):
                new_data_to_add_by_type = all_data[j]
                role_by_data_type = dict_grouped_by_role[data_type]
                add_data_to = role_by_data_type.setdefault(role_key, [])
                add_data_to.extend(new_data_to_add_by_type)

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
    '''Helper function to get_individual_ideal_peak. Should be used directly.'''
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
            
            
            # Add the zero arrays to the current trimmed peak
            trimmed_peak = np.concatenate([strat_zeros, peak_data[curr_inteval[trim_strat_interval_index]:curr_inteval[trim_end_interval_index]], end_zeros])
            trimmed_peaks.append(trimmed_peak)


        # Average the trimmed peaks
        ideal_peak = np.mean(trimmed_peaks, axis=0)
        # Save the ideal peak
        ideal_peaks[noise_maker_key] = ideal_peak


    return ideal_peaks


def convolve_ideal_peak_over_filtered_data(final_dict, ideal_peaks, by='by_receiver'):
    '''
    Description
    ------------
    This function will convolve the ideal peaks over the peaks of the filtered data of the individuals and plot the results.

    Parameters
    ------------
    final_dict : dict
        The dictionary returned by the combined_wavs_by_communicators function
    ideal_peaks : dict
        The dictionary returned by the get_individual_ideal_peak function
    by : str
        The key to the dictionary in final_dict to use, either 'by_maker', 'by_receiver' or 'by_both'
    
    Returns
    ------------
    None    
    '''
    
    data_dict = final_dict[by]
    data_keys = data_dict['grouped_data'].keys()
    
    peak_keys = ideal_peaks.keys()
    convolution_results = {peak_key : {} for peak_key in peak_keys}

    for peak_individual in peak_keys:
        peak = ideal_peaks[peak_individual]
        curr_dict = convolution_results[peak_individual]
        
        for data_individual in data_keys:
            filtered_data_peaks = data_dict['grouped_peaks'][data_individual]

            for i, sample in enumerate(filtered_data_peaks):
                matched_filter_output = convolve(sample, peak, mode='same')

                if data_individual not in curr_dict.keys():
                    curr_dict[data_individual] = [np.max(matched_filter_output)]
                else:
                    curr_dict[data_individual].append(np.max(matched_filter_output))
    
    fig, axes = plt.subplots(len(peak_keys), 1, figsize=(10,15))
    plt.subplots_adjust(hspace=0.6)
    colors = ['indianred', 'darkorange', 'teal', 'royalblue', 'indigo']

    for i, peak_ind in enumerate(peak_keys):
        ax = axes[i]
        ax.hist(convolution_results[peak_ind][peak_ind], bins=20, color=colors[i], alpha=0.5)
        ax.set_title(f'Convolution score histogram of {peak_ind}', fontsize=16)

        ax.set_xlabel('Convolution Score', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()  


def plot_peak_on_grouped_data(final_dict, sample_rate=44100, by='by_both', n_rows=3):
    '''
    Description
    ------------
    This function will plot the grouped data of the individuals along with the idealized peaks.

    Parameters
    ------------
    final_dict : dict
        The dictionary returned by the combined_wavs_by_communicators function
    sample_rate : int
        The sample rate of the data
    by : str
        The key to the dictionary in final_dict to use, either 'by_maker', 'by_receiver' or 'by_both'
    n_rows : int
        The number of samples to plot

    Returns
    ------------
    None    
    '''
    data_dict = final_dict[by]
    keys = data_dict['grouped_data'].keys()
    dt = 1/sample_rate

    for i, individual in enumerate(keys):
        n_samples = min(n_rows, len(data_dict['grouped_data'][individual]))
        fig, axes = plt.subplots(n_samples, 2, figsize=(16,10))
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        raw_data = data_dict['grouped_data'][individual]
        filtered_data = data_dict['filtered_data'][individual]
        peaks = data_dict['grouped_peaks'][individual]
        peak_intervals = data_dict['grouped_intervals'][individual]

        axes[0,0].set_title('Unfiltered Signal', fontsize=18)
        axes[0,1].set_title('Filtered Signal', fontsize=18)

        for sample in range(n_samples):

            ax = axes[sample, 0]
            ax.plot(np.arange(0, raw_data[sample].shape[-1])*dt, raw_data[sample], c='cornflowerblue')
            ax.plot(peak_intervals[sample]*dt, peaks[sample], c='mediumvioletred', lw=4, alpha=0.9, ls='--')
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.set_ylabel('Amplitude (Voltage)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            ax = axes[sample, 1]
            ax.plot(np.arange(0, raw_data[sample].shape[-1])*dt, filtered_data[sample], c='cornflowerblue')
            ax.plot(peak_intervals[sample]*dt, peaks[sample], c='mediumvioletred', lw=4, alpha=0.9, ls='--')
            ax.set_xlabel('Time (s)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

        fig.suptitle(individual, fontsize=20)
        axes[0,1].legend(['data', 'idealized peak'], loc='best', fontsize=13)
        plt.show()


def convolution_results_hist_vals(grouped_peaks_dict, ideal_peaks):
    '''
    Description
    ------------
    This function will convolve the ideal peaks over the peaks of the filtered data of the individuals and return the convolution results.

    Parameters
    ------------
    grouped_peaks_dict : dict
        The dictionary returned by the combined_wavs_by_communicators function
    ideal_peaks : dict
        The dictionary returned by the get_individual_ideal_peak function
    
    Returns
    ------------
    convolution_results : dict
        A dictionary with the convolution results
    '''
    peak_keys = ideal_peaks.keys()
    convolution_results = {peak_key : {} for peak_key in peak_keys}

    for data_ind, data_peaks in grouped_peaks_dict.items():
        
        for peak_ind, peak in ideal_peaks.items():
            curr_dict = convolution_results[peak_ind]
            
            for i, sample in enumerate(data_peaks):
                matched_filter_output = convolve(sample, peak, mode='same')

                if data_ind not in curr_dict.keys():
                    curr_dict[data_ind] = [np.max(matched_filter_output)]
                else:
                    curr_dict[data_ind].append(np.max(matched_filter_output))
    
    return convolution_results


def make_distribution(train, test, by='by_maker', peak_width_determination_method='mean', file_list=False, sigma=20):
    '''
    Description
    ------------
    This function will combine the WAV files of the train and test sets and then calculate the convolution results of the test set over the train set.

    Parameters
    ------------
    train : list
        A list of the paths to the folders containing the WAV files for the train set.
    test : list
        A list of the paths to the folders containing the WAV files for the test set.
    by : str
        The key to the dictionary in final_dict to use, either 'by_maker', 'by_receiver' or 'by_both'
    peak_width_determination_method : str
        The method used to calculate the ideal peak width, either 'mean', 'median' or 'min'. See get_individual_ideal_peak for more information.
    
    Returns
    ------------
    train_convolution_results : dict
        A dictionary with the convolution results of the train set
    test_convolution_results : dict
        A dictionary with the convolution results of the test set
    '''
    train_dict = combine_wavs_by_communicators(train, sigma=sigma, file_list=file_list)
    test_dict = combine_wavs_by_communicators(test, sigma=sigma, file_list=file_list)
    
    ideal_peaks = __find_ideal_peaks(train_dict[by]['filtered_data'], train_dict[by]['grouped_intervals'], peak_width_determination_method)
    test_peaks_dict = test_dict[by]['grouped_peaks']

    train_convolution_results = convolution_results_hist_vals(train_dict[by]['grouped_peaks'], ideal_peaks)
    test_convolution_results = convolution_results_hist_vals(test_peaks_dict, ideal_peaks)
    
    return train_convolution_results, test_convolution_results


def split_test_train_folders(folders_to_iter, n_tests):
    '''
    Description
    ------------
    This function will split the folders into test and train sets.

    Parameters
    ------------
    folders_to_iter : list
        A list of the paths to the folders containing the WAV files.
    n_tests : int
        The number of test files to use.
    
    Returns
    ------------
    test_data_by_makers : list
        A list of the paths to the test files by noise maker.
    train_data_by_makers : list
        A list of the paths to the train files by noise maker.
    test_data_by_receivers : list
        A list of the paths to the test files by noise receiver.
    train_data_by_receivers : list
        A list of the paths to the train files by noise receiver.
    '''
    labels_by_makers = []
    labels_by_receivers = []
    
    for j, folder in enumerate(folders_to_iter):
        # Split folder path using underscores
        parts = os.path.basename(folder).split('_')
        
        # Extract individuals' names
        noise_maker = parts[0]
        noise_receiver = parts[2]

        if noise_maker not in labels_by_makers:
            labels_by_makers.append(noise_maker[:4])
        if noise_receiver[:4] not in labels_by_receivers:
            labels_by_receivers.append(noise_receiver[:4])

        if len(labels_by_makers) == 4 and len(labels_by_receivers) == 5:
            break
 
    n_files_per_ind_by_maker = n_tests // len(labels_by_makers)
    n_files_per_ind_by_receiver = n_tests // len(labels_by_receivers)

    n_files_by_makers = [n_files_per_ind_by_maker if i != len(labels_by_makers)-1 else (n_tests - n_files_per_ind_by_maker * (len(labels_by_makers) - 1)) for i in range(len(labels_by_makers))]
    n_files_by_receiver = [n_files_per_ind_by_receiver if i != len(labels_by_receivers)-1 else (n_tests - n_files_per_ind_by_receiver * (len(labels_by_receivers) - 1)) for i in range(len(labels_by_receivers))]
    
    final_dict = combine_wavs_by_communicators(folders_to_iter)

    test_data_by_makers = [] 
    test_data_by_receivers = [] 

    train_data_by_makers = [] 
    train_data_by_receivers = []

    for i, maker in enumerate(labels_by_makers):
        n = n_files_by_makers[i]
        test_data_by_makers.extend(final_dict['by_maker']['file_names'][maker][:n])
        train_data_by_makers.extend(final_dict['by_maker']['file_names'][maker][n:])

    for i, receiver in enumerate(labels_by_receivers):
        n = n_files_by_receiver[i]
        test_data_by_receivers.extend(final_dict['by_receiver']['file_names'][receiver][:n])
        train_data_by_receivers.extend(final_dict['by_receiver']['file_names'][receiver][n:])
    
    return test_data_by_makers, train_data_by_makers, test_data_by_receivers, train_data_by_receivers


def percentiles(train_convolution_results, test_convolution_results, ideal_peaks, percentile_threshold=90):
    '''
    Description
    ------------
    This function will calculate the fraction of the test convolution scores that are above a certain percentile threshold of the train convolution scores
    of an individual with itself.

    Parameters
    ------------
    train_convolution_results : dict
        A dictionary with the convolution results of the train set
    test_convolution_results : dict
        A dictionary with the convolution results of the test set
    ideal_peaks : dict
        The dictionary returned by the get_individual_ideal_peak function
    percentile_threshold : int
        The percentile threshold to use
    
    Returns
    ------------
    tested_inds_accuracies : dict
        A dictionary with the fraction of the test convolution scores that are above a certain percentile threshold of the train convolution scores
    '''
    tested_inds_accuracies = {tested_ind : {peak_ind : 0 for peak_ind in ideal_peaks.keys()} for tested_ind in test_convolution_results.keys()}
    for train_ind, convolution_scores_dict in train_convolution_results.items():
        for peak_ind in ideal_peaks.keys():
            value_threshold = np.percentile(np.array(train_convolution_results[peak_ind][peak_ind]), percentile_threshold)
            test_ind_array = np.array(test_convolution_results[train_ind][peak_ind])
            test_bigger_than_threshold = np.mean(test_ind_array > value_threshold)
            tested_inds_accuracies[train_ind][peak_ind] = test_bigger_than_threshold
    
    return tested_inds_accuracies


if __name__ == '__main__':
    main()