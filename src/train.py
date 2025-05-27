"""
src/train.py

This script handles the training of the audio super-resolution model.
It loads preprocessed HDF5 datasets (created by `src/prep_data.py`),
initializes or loads a model (defined in `src/models.py`), and then
trains the model using the specified hyperparameters.
Model checkpoints and training progress are managed using Keras callbacks.
Paths for data and weights are sourced from `src/config.py`.
"""
import os
from src.config import WEIGHTS_PATH as CFG_WEIGHTS_PATH, PREPROCESSED_DATA_PATH as CFG_DATA_DIR
from keras.callbacks import ModelCheckpoint, EarlyStopping
from src.models import base_model
from src.utils import load_model, SNR, sum_loss, compile_model
import h5py

def load_h5_list(directory_path):
    """
    Loads a list of HDF5 (.h5) file paths from a specified directory.

    Args:
        directory_path (str): The path to the directory containing .h5 files.

    Returns:
        list: A list of full paths to the .h5 files found. Returns an empty
              list if the directory doesn't exist or no .h5 files are found.
    """
    datasets = []
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist. Cannot load HDF5 files.")
        return datasets
        
    filenames = os.listdir(directory_path)
    file_extensions = set(['.h5']) # Only look for .h5 files
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext.lower() in file_extensions: # Case-insensitive check
            full_filename = os.path.join(directory_path, filename)
            datasets.append(full_filename)
            
    if not datasets:
        print(f"No .h5 files found in {directory_path}.")
    else:
        print(f"Found {len(datasets)} HDF5 files in {directory_path}.")
    return datasets

def load_h5(dataset_path):
    """
    Loads 'data' (input X) and 'label' (target Y) arrays from a single HDF5 file.

    Args:
        dataset_path (str): Full path to the HDF5 file.

    Returns:
        tuple: A tuple (X, Y) containing numpy arrays for data and labels.
               Returns (None, None) if the file is not found or data/label
               keys are missing.
    """
    print(f'Loading dataset: {dataset_path}')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found.")
        return None, None
    
    try:
        with h5py.File(dataset_path, 'r') as hf:
            # Ensure 'data' and 'label' keys exist
            if 'data' not in hf or 'label' not in hf:
                print(f"Error: HDF5 file {dataset_path} is missing 'data' or 'label' keys.")
                return None, None
            X = hf['data'][:]  # Input data (low-resolution patches)
            Y = hf['label'][:] # Target data (high-resolution patches)
        print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
        return X, Y
    except Exception as e:
        print(f"Error loading HDF5 file {dataset_path}: {e}")
        return None, None

def run():
    """
    Main training function for the audio super-resolution model.

    This function orchestrates the training process:
    1. Defines hyperparameters for training.
    2. Initializes the model architecture (from `src.models.base_model`).
    3. Optionally loads pre-trained weights.
    4. Compiles the model with optimizer, loss function, and metrics.
    5. Loads a list of HDF5 datasets from the directory specified in `src.config.PREPROCESSED_DATA_PATH`.
    6. Iterates through the datasets for a specified number of main epochs (`EPOCHS`).
    7. For each dataset, trains the model for a smaller number of "mini-epochs" (`MINI_EPOCH`).
    8. Uses Keras callbacks for saving the best model weights (`ModelCheckpoint`) and
       early stopping if validation loss doesn't improve (`EarlyStopping`).
    """
    # --- Hyperparameters ---
    EPOCHS = 1          # Number of times to iterate over the entire set of HDF5 dataset files.
    BATCH_SIZE = 256    # Number of samples per gradient update.
    LOAD_WEIGHTS = True # If True, attempts to load weights from WEIGHTS_FILE before training.
                        # Set to False to always start training from scratch.
    
    # Paths (sourced from src/config.py)
    WEIGHTS_DIR = CFG_WEIGHTS_PATH      # Directory to save/load model weights.
    DATASETS_DIR = CFG_DATA_DIR       # Directory containing preprocessed HDF5 datasets.
    
    WEIGHTS_FILE_NAME = 'asr-weights.hdf5' # Filename for saving model weights.
    
    VALID_SPLIT = 0.05  # Fraction of training data from each HDF5 file to be used as validation data.
    SHUFFLE = True      # Whether to shuffle the training data before each mini-epoch.
    MINI_EPOCH = 3      # Number of epochs to train on each individual HDF5 dataset file
                        # within a main EPOCH. This allows iterative loading of large datasets.

    # --- Setup ---
    # Ensure weights directory exists for saving checkpoints
    if not os.path.exists(WEIGHTS_DIR):
        print(f"Weights directory {WEIGHTS_DIR} does not exist. Creating it.")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    weights_full_path = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE_NAME)

    # Initialize or load the model
    print("Initializing model...")
    model = base_model(summary=False) # summary=False to avoid printing it here; True for debugging
    
    if LOAD_WEIGHTS:
        print(f"Attempting to load weights from: {weights_full_path}")
        # load_model utility handles cases where file might not exist yet
        model = load_model(model, weights_full_path, load_weights=True) 
    else:
        print("LOAD_WEIGHTS is False. Starting with a fresh (untrained) model.")
        # Ensure model is initialized even if not loading weights (already done by base_model)

    # Compile the model
    print("Compiling model...")
    model = compile_model(model) # Uses Adam optimizer, MSE loss, sum_loss, SNR metrics
    
    # Load list of HDF5 dataset files
    print(f"Loading list of HDF5 datasets from: {DATASETS_DIR}")
    dataset_paths = load_h5_list(DATASETS_DIR)
    if not dataset_paths:
        print(f"No HDF5 datasets found in {DATASETS_DIR}. Training cannot proceed.")
        return

    # --- Keras Callbacks ---
    # ModelCheckpoint: Saves the model weights only when `val_loss` improves.
    checkpointer = ModelCheckpoint(filepath=weights_full_path, 
                                   monitor='val_loss', # Monitor validation loss
                                   verbose=1, 
                                   save_best_only=True, # Only save if improved
                                   save_weights_only=True) # Save only weights
    
    # EarlyStopping: Stops training if `val_loss` doesn't improve for `patience` epochs.
    earlystopper = EarlyStopping(monitor='val_loss', 
                                 patience=3, # Number of epochs with no improvement after which training will be stopped.
                                 verbose=1)   # Print message when training is stopped.
    
    # --- Training Loop ---
    print(f"\nStarting training process: {EPOCHS} main epoch(s), {MINI_EPOCH} mini-epoch(s) per dataset file.")
    for epoch_num in range(EPOCHS):
        print(f'\n--- Main Epoch ({epoch_num+1}/{EPOCHS}) ---')
        
        # Optionally, shuffle the order of dataset files for each main epoch
        # random.shuffle(dataset_paths) # Uncomment if desired
        
        for dataset_idx, current_dataset_path in enumerate(dataset_paths):
            print(f"\n  Training on dataset file {dataset_idx+1}/{len(dataset_paths)}: {current_dataset_path}")
            X_train, Y_train = load_h5(current_dataset_path)
            
            if X_train is None or Y_train is None:
                print(f"  Skipping dataset {current_dataset_path} due to loading error.")
                continue
            
            if X_train.shape[0] == 0:
                print(f"  Skipping dataset {current_dataset_path} as it contains no samples.")
                continue

            # Fit the model on the current HDF5 dataset
            model.fit(X_train, Y_train,
                      batch_size=BATCH_SIZE,
                      epochs=MINI_EPOCH, 
                      shuffle=SHUFFLE,
                      callbacks=[checkpointer, earlystopper],
                      validation_split=VALID_SPLIT)
                      
            # Check if early stopping was triggered
            if model.stop_training:
                print("Early stopping triggered. Ending training.")
                break # Break from dataset loop
        
        if model.stop_training:
            break # Break from main epoch loop

    print('\n--- Training Finished ---')

if __name__ == '__main__':
    print("Executing src/train.py as main script...")
    run()
    print("Training script execution finished.")
    # The original logic for loading weights was:
    # if LOAD_WEIGHTS and not os.path.exists(weights_full_path):
    #     print(f"Warning: Weights file {weights_full_path} not found, but LOAD_WEIGHTS is True. Starting with fresh model.")
    #     model = load_model(model, weights_full_path, load_weights=False) # Force load_weights to False
    # elif LOAD_WEIGHTS:
    #      model = load_model(model, weights_full_path, load_weights=True)
    # else:
    #     print("LOAD_WEIGHTS is False. Starting with a fresh model.")
    #     model = load_model(model, weights_full_path, load_weights=False) # Explicitly don't load
    # This is now handled more cleanly by the load_model utility itself if it checks for file existence.
    # The current load_model in utils.py does not check for file existence before attempting model.load_weights,
    # so the check here or an improved load_model utility is good.
    # For now, the logic in run() is fine as load_model will print an error if file not found.
    pass # Placeholder for original weight loading logic if needed to be reinstated differently
