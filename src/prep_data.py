"""
src/prep_data.py

This script is responsible for preparing audio data for the super-resolution model.
It includes functionalities for:
- Loading WAV audio files from a specified corpus directory.
- Preprocessing audio data by:
    - Loading high-resolution audio.
    - Generating a low-resolution version by direct downsampling.
    - Upsampling the low-resolution version using spline interpolation to match the
      dimensions of the high-resolution signal (this interpolated version serves as model input).
    - Extracting corresponding patches from both the interpolated low-resolution
      and the original high-resolution audio.
- Saving these patches into HDF5 files for efficient loading during model training.
The paths for data input (corpus) and output (HDF5 files) are configured in `src/config.py`.
"""
import os
import numpy as np
import librosa
from src.utils import upsample # For spline interpolation
import h5py
import random 
from itertools import chain 
from src.config import VCTK_CORPUS_PATH, PREPROCESSED_DATA_PATH

# my_wav_dir is set to the VCTK_CORPUS_PATH from config, which is the root directory for WAV files.
my_wav_dir = VCTK_CORPUS_PATH 

def save(dataset_name, X_data, Y_data):
    """
    Saves the low-resolution (X) and high-resolution (Y) patch datasets to an HDF5 file.

    Args:
        dataset_name (str): The full path and filename for the HDF5 file.
        X_data (np.ndarray): The dataset of low-resolution (input) patches.
        Y_data (np.ndarray): The dataset of high-resolution (target) patches.
    """
    # Ensure the directory for the dataset exists
    os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
    
    with h5py.File(dataset_name, 'w') as f:
        # 'data' dataset stores the low-resolution (input) patches
        data_set = f.create_dataset('data', X_data.shape, np.float32) 
        # 'label' dataset stores the high-resolution (target) patches
        label_set = f.create_dataset('label', Y_data.shape, np.float32)
        data_set[...] = X_data
        label_set[...] = Y_data
    print(f'Save complete -> {dataset_name}')

def preprocess(file_list, start_index, end_index, target_sr=48000, downsample_scale=6, patch_dimension=64, patch_stride=8, tag='train'):
    """
    Preprocesses a list of audio files to extract low-resolution and high-resolution patches.

    For each audio file:
    1. Loads the high-resolution (HR) audio.
    2. Creates a low-resolution (LR) version by direct downsampling (slicing).
    3. Upsamples the LR version using spline interpolation to match HR dimensions; this becomes the model input.
    4. Extracts patches of `patch_dimension` with a given `patch_stride` from both the
       interpolated LR audio (model input) and the original HR audio (target).

    Args:
        file_list (list): List of full paths to WAV audio files.
        start_index (int): Starting index in `file_list` to process.
        end_index (int): Ending index in `file_list` to process.
        target_sr (int, optional): The target sample rate to load audio files. Defaults to 48000.
        downsample_scale (int, optional): Factor by which to downsample HR audio to create LR audio.
                                     Also the factor for upsampling LR to match HR length. Defaults to 6.
        patch_dimension (int, optional): The length (in samples) of each audio patch. Defaults to 64.
        patch_stride (int, optional): The step size (in samples) to move when extracting patches. Defaults to 8.
        tag (str, optional): A tag (e.g., 'train', 'test') to include in the output HDF5 filename. Defaults to 'train'.

    Returns:
        tuple: A tuple containing:
            - lr_patches_all (np.ndarray): Array of all extracted low-resolution (input) patches.
            - hr_patches_all (np.ndarray): Array of all extracted high-resolution (target) patches.
            - dataset_name (str): The generated name/path for the HDF5 dataset file.
    """
    # Shuffle the segment of the file list to be processed for randomness in patch generation
    # This shuffles a sub-list in place if file_list is mutable, or a copy if it's not.
    # To be safe and explicit, let's operate on a sub_list
    process_list = file_list[start_index : end_index + 1]
    random.shuffle(process_list) 
    
    num_files_to_process = len(process_list)
    
    lr_patches_collected = list()
    hr_patches_collected = list()
    
    for i, wav_path in enumerate(process_list):
        # Progress indication: current_file_index_in_list is used for overall progress if start_index > 0
        # For this loop, 'i' is the index within 'process_list'
        if i % 10 == 0 : 
            print(f"Processing file {wav_path} ({i+1}/{num_files_to_process} in current batch, overall index {start_index + i})")

        # Load high-resolution (HR) audio
        x_hr, fs = librosa.load(wav_path, sr=target_sr)
        
        # Ensure audio length is a multiple of the downsample_scale for clean downsampling
        x_len = len(x_hr)
        x_hr = x_hr[ : x_len - (x_len % downsample_scale)]
        
        # Create Low-Resolution (LR) version by direct downsampling (array slicing)
        x_lr_native = np.array(x_hr[0::downsample_scale])
        
        # Upsample the native LR version using spline interpolation to match HR length.
        # This interpolated version becomes the input (X) to the model.
        x_lr_interpolated = upsample(x_lr_native, downsample_scale)
        
        # Reshape to (length, 1) for consistency, as model might expect a channel dimension
        x_lr_interpolated = np.reshape(x_lr_interpolated, (len(x_lr_interpolated), 1))
        x_hr = np.reshape(x_hr, (len(x_hr), 1)) # Original HR is the target (Y)
        
        # Extract patches
        for patch_idx in range(0, x_lr_interpolated.shape[0] - patch_dimension + 1, patch_stride):
            lr_patch = x_lr_interpolated[patch_idx : patch_idx + patch_dimension]
            hr_patch = x_hr[patch_idx : patch_idx + patch_dimension] # Corresponding HR patch
            
            lr_patches_collected.append(lr_patch)
            hr_patches_collected.append(hr_patch)
    
    lr_patches_all = np.array(lr_patches_collected)
    hr_patches_all = np.array(hr_patches_collected)
    
    print(f'Shape of high-resolution (Y) patches: {hr_patches_all.shape}')
    print(f'Shape of low-resolution (X) patches: {lr_patches_all.shape}')
                
    # Construct dataset filename using PREPROCESSED_DATA_PATH from config
    filename_template = 'asr-ex%d-start%d-end%d-scale%d-sr%d-dim%d-strd%d-%s.h5'
    actual_num_files_processed = num_files_to_process # Renamed data_size for clarity
    filename = filename_template % (
        actual_num_files_processed, start_index, end_index, 
        downsample_scale, target_sr, patch_dimension, patch_stride, tag
    )
    dataset_name = os.path.join(PREPROCESSED_DATA_PATH, filename)
    
    print(f"Preprocessing complete. Generated dataset name: {dataset_name}")
    return lr_patches_all, hr_patches_all, dataset_name

def load_wav_list_for_test(base_directory):
    """
    Loads a list of .wav file paths from a single directory (non-recursive).
    Typically used for loading test samples where files are flat in one folder.

    Args:
        base_directory (str): The path to the directory containing .wav files.

    Returns:
        list: A list of full paths to the .wav files found in the directory.
              Returns an empty list if the directory is not found or contains no .wav files.
    """
    file_list = []    
    if not os.path.isdir(base_directory):
        print(f"Error: Test samples directory '{base_directory}' not found.")
        return file_list
        
    filenames = os.listdir(base_directory)
    file_extensions = set(['.wav']) # Case-sensitive, use .lower() for case-insensitivity if needed
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext.lower() in file_extensions: # Make check case-insensitive
            full_filename = os.path.join(base_directory, filename)
            file_list.append(full_filename)
    
    if file_list:
        print(f'Found {len(file_list)} .wav files in {base_directory}. First few examples:')
        for i, file_path in enumerate(file_list):
            print(file_path)
            if i >= 4: # Print max 5 examples
                break
    else:
        print(f"No .wav files found in {base_directory}.")

    return file_list

def load_wav_list(corpus_base_dir):
    """
    Loads a list of .wav file paths from a corpus directory, assuming a specific
    subdirectory structure (e.g., VCTK-like: corpus_base_dir/speaker_id/wav_file.wav).

    The function iterates through subdirectories (assumed to be speaker IDs) within
    `corpus_base_dir`, collects all .wav files from these subdirectories, and
    constructs their full paths. It also attempts to parse speaker IDs from filenames
    (e.g., 'p<speaker_id>_*.wav') to correctly form the path.

    Args:
        corpus_base_dir (str): The root directory of the speech corpus.

    Returns:
        list: A list of full paths to all .wav files found within the assumed
              speaker subdirectories. Returns an empty list if the base directory
              is not found or no suitable .wav files are found.
    """
    wav_paths_list = []
    if not os.path.isdir(corpus_base_dir):
        print(f"Error: Corpus directory '{corpus_base_dir}' not found.")
        return wav_paths_list

    speaker_dirs = os.listdir(corpus_base_dir)
    
    for speaker_dir_name in speaker_dirs:
        current_speaker_path = os.path.join(corpus_base_dir, speaker_dir_name)
        if not os.path.isdir(current_speaker_path):
            # Skip if it's a file, not a speaker directory
            continue

        wav_files_in_speaker_dir = []
        for f_name in os.listdir(current_speaker_path):
            if f_name.lower().endswith('.wav'):
                 wav_files_in_speaker_dir.append(f_name)
        
        for wav_filename in wav_files_in_speaker_dir:
            # This part assumes a VCTK-like structure where the subdirectory name (speaker_dir_name)
            # is the speaker ID, and is also part of the path.
            # The original code had a line: `fir, throwaway = wav_list_file.split(sep='_')`
            # and then constructed `sss = my_wav_dir + '/' + fir + '/' + wav_list_file`.
            # This implies `fir` (the part before '_') should match `speaker_dir_name`.
            # We will simplify this: the path is corpus_base_dir/speaker_dir_name/wav_filename.
            
            # Optional: Validate if wav_filename prefix matches speaker_dir_name if that's an invariant.
            # try:
            #     filename_prefix = wav_filename.split('_')[0]
            #     if filename_prefix != speaker_dir_name: # Or some transformation of it like 'p' + speaker_id
            #         print(f"Warning: Filename prefix '{filename_prefix}' does not match speaker directory '{speaker_dir_name}' for {wav_filename}. Using directory name for path.")
            # except IndexError:
            #     print(f"Warning: Could not parse prefix from filename '{wav_filename}'. Using directory name for path.")

            full_wav_path = os.path.join(current_speaker_path, wav_filename) 
            wav_paths_list.append(full_wav_path)
    
    if not wav_paths_list:
        print(f"No .wav files found in subdirectories of {corpus_base_dir} following expected structure.")
    else:
        print(f"Loaded {len(wav_paths_list)} wav files from {corpus_base_dir}.")
    return wav_paths_list

def run():
    """
    Main execution function for data preparation.

    This function orchestrates the loading of WAV file paths from the corpus,
    iteratively preprocesses them in batches (defined by `dataset_size` and `convert_limit`),
    and saves the resulting low-resolution and high-resolution patches into HDF5 files.
    Configuration parameters for this process (e.g., sample rates, patch dimensions, limits)
    are defined within this function. Paths are sourced from `src/config.py`.
    """
    # --- Configuration for data preparation ---
    convert_limit = 10  # Maximum number of files to process from the file_list (e.g., for quick testing).
                        # Set to a very large number or len(file_list) to process all.
    dataset_size  = 10  # Number of audio files to process per HDF5 dataset.
    
    # Audio processing parameters (should match model expectations)
    sr_target = 48000          # Target sampling rate for loading audio.
    scale_factor = 8           # Downscaling factor for creating LR audio.
    patch_dim = 256     # Dimension (length) of audio patches.
    patch_stride_val = 64         # Stride for extracting patches.
    
    # Ensure VCTK_CORPUS_PATH (aliased as my_wav_dir) is a valid directory
    if not os.path.isdir(my_wav_dir): 
        print(f"Error: VCTK_CORPUS_PATH ('{my_wav_dir}') is not a valid directory. Please check src/config.py.")
        return

    print(f"Loading audio file list from corpus: {my_wav_dir}")
    all_audio_files = load_wav_list(base_dir=my_wav_dir)
    
    if not all_audio_files:
        print(f"No WAV files found in the corpus directory '{my_wav_dir}' or its subdirectories. Exiting.")
        return
    
    # Adjust convert_limit if it's larger than the number of available files
    actual_convert_limit = min(convert_limit, len(all_audio_files))
    if convert_limit > len(all_audio_files):
        print(f"Note: convert_limit ({convert_limit}) is greater than total files found ({len(all_audio_files)}). Processing all available files up to {actual_convert_limit}.")

    print(f"Starting preprocessing. Will process up to {actual_convert_limit} files in batches of {dataset_size}.")
    for i in range(0, actual_convert_limit, dataset_size):
        # Determine the end index for the current batch
        # Ensure it doesn't exceed the actual_convert_limit or the total number of files
        current_batch_end_index = min(i + dataset_size - 1, actual_convert_limit - 1)
        
        # The 'start' and 'end' for preprocess are indices into all_audio_files list
        print(f"\nProcessing batch: files from index {i} to {current_batch_end_index}")
        
        # Call preprocess for the current batch of files
        # Note: preprocess shuffles its input sub-list internally
        train_X, train_Y, trainset_hdf5_name = preprocess(
            file_list=all_audio_files,
            start_index=i, 
            end_index=current_batch_end_index,
            target_sr=sr_target, 
            downsample_scale=scale_factor,
            patch_dimension=patch_dim,
            patch_stride=patch_stride_val,
            tag='train'  # Tag for training data
        )
        
        # Save the generated patches to an HDF5 file
        # The directory for trainset_hdf5_name (PREPROCESSED_DATA_PATH) is handled by save()
        if train_X.shape[0] > 0 and train_Y.shape[0] > 0: # Check if any patches were actually created
            save(trainset_hdf5_name, train_X, train_Y)
        else:
            print(f"No patches generated for batch starting at index {i}. Skipping save for {trainset_hdf5_name}")

    print("\nData preparation script finished.")

if __name__ == '__main__':
    print("Running data preparation script...")
    run()
    # lr_len = len(lr_patches_collected) # Not strictly needed as np.array handles list directly
    
    # hr_patches_all = np.array(hr_patches_collected) # Renamed from hr_patches
    # lr_patches_all = np.array(lr_patches_collected) # Renamed from lr_patches
    
    # print(f'Shape of high-resolution (Y) patches: {hr_patches_all.shape}') # Moved to preprocess
    # print(f'Shape of low-resolution (X) patches: {lr_patches_all.shape}') # Moved to preprocess
                
    # # Path for dataset_name constructed using PREPROCESSED_DATA_PATH from config
    # filename_template = 'asr-ex%d-start%d-end%d-scale%d-sr%d-dim%d-strd%d-%s.h5'
    # filename = filename_template % (data_size, start, end, scale, sr, dimension, stride, tag)
    # dataset_name = os.path.join(PREPROCESSED_DATA_PATH, filename) # Moved to preprocess
    
    # print("preprocess function works") # Changed from "works" to be more descriptive # Moved to preprocess
    # return lr_patches_all, hr_patches_all, dataset_name # Moved to preprocess

# def load_wav_list_for_test(base_dir): # Definition moved up
#     pass

# def load_wav_list(base_dir): # Definition moved up
#     pass

# def run(): # Definition moved up
#     pass

if __name__ == '__main__':
    print("Running data preparation script...")
    run()
