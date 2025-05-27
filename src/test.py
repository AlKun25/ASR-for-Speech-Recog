"""
src/test.py

This script is used to test the trained audio super-resolution model.
It processes sample WAV files from a specified test directory by:
1. Loading the trained model weights.
2. Reading WAV files from the `TEST_SAMPLES_PATH` (configured in `src/config.py`).
3. Preprocessing these audio files into low-resolution (input) and
   high-resolution (ground truth) patches suitable for the model.
4. Performing inference with the model to generate super-resolved audio.
5. Evaluating the model's performance using metrics like SNR.
6. Saving the original high-resolution, a true downsampled low-resolution version (example),
   and the super-resolved output audio as WAV files in `OUTPUT_AUDIO_PATH`.
7. Optionally, displaying STFT spectrum plots for these audio versions.
"""
import os
from src.config import WEIGHTS_PATH as CFG_WEIGHTS_PATH, TEST_SAMPLES_PATH as CFG_TEST_SAMPLES_PATH, OUTPUT_AUDIO_PATH as CFG_OUTPUT_AUDIO_PATH
import numpy as np
from src.models import base_model
from src.utils import load_model, compile_model, STFT
from src.prep_data import preprocess, load_wav_list_for_test
import librosa
import soundfile as sf

# Imports from the original notebook that are not used in this script:
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# import h5py

def test_samples():
    """
    Tests the trained audio super-resolution model on sample WAV files.

    The process involves:
    - Defining configuration parameters (sample rates, scaling factors, patch dimensions).
    - Loading model weights from the path specified in `src/config.py`.
    - Loading test WAV files from `src/config.TEST_SAMPLES_PATH`.
    - Preprocessing audio into patches:
        - `X_lr_interpolated`: Low-resolution audio, upsampled by interpolation (model input).
        - `Y_hr`: Original high-resolution audio (ground truth).
    - Predicting super-resolved audio (`pred_hr`) from `X_lr_interpolated`.
    - Evaluating the prediction against `Y_hr` using model metrics.
    - Generating STFT plots for visual comparison of original HR, true LR (example),
      interpolated LR (model input), and super-resolved HR.
    - Saving the original HR, a true LR example, and the super-resolved HR audio
      as WAV files to `src/config.OUTPUT_AUDIO_PATH`.

    Key Configuration Parameters (hardcoded or from `src/config.py`):
    - SR: Target sample rate for high-resolution audio.
    - SCALE: Downsampling/upsampling factor.
    - DIMENSION: Patch size for model input/output.
    - STRIDE: Stride for patch extraction (typically equals DIMENSION for non-overlapping test patches).
    - WEIGHTS_PATH, WEIGHTS_FILE: Location of trained model weights.
    - TEST_SAMPLES_DIR: Directory containing input WAV files for testing.
    - OUTPUT_AUDIO_DIR: Directory where output WAV files will be saved.
    """
    # --- Configuration Parameters ---
    # These parameters should align with those used during model training and data preparation.
    SR = 48000  # Standard Sample Rate for high-resolution audio
    SCALE = 6   # The factor by which audio was downsampled/upsampled
    DIMENSION = 256 # The dimension of input/output patches for the model
    STRIDE = 256    # Stride used for creating patches; for testing, DIMENSION ensures non-overlapping patches

    DOWNSAMPLED_SR = SR // SCALE # Calculated sample rate for low-resolution audio (e.g., 8000 Hz)

    # --- File and Directory Paths (from src/config.py) ---
    LOAD_WEIGHTS = True  # Flag to load pre-trained model weights
    WEIGHTS_PATH = CFG_WEIGHTS_PATH
    WEIGHTS_FILE = 'asr-weights.hdf5'  # Specific model weights filename
    TEST_SAMPLES_DIR = CFG_TEST_SAMPLES_PATH
    OUTPUT_AUDIO_DIR = CFG_OUTPUT_AUDIO_PATH

    # --- 1. Model Loading and Compilation ---
    print("--- 1. Model Loading and Compilation ---")
    # Validate existence of weights directory and file
    if not os.path.isdir(WEIGHTS_PATH):
        print(f"Error: Weights directory '{WEIGHTS_PATH}' not found. Please ensure it exists.")
        return
    weights_full_path = os.path.join(WEIGHTS_PATH, WEIGHTS_FILE)
    if not os.path.exists(weights_full_path):
        print(f"Error: Weights file '{weights_full_path}' not found in '{WEIGHTS_PATH}'.")
        return

    print("Initializing and compiling model...")
    model = base_model(summary=False)  # summary=False to prevent printing the model architecture
    model = load_model(model, weights_full_path, load_weights=LOAD_WEIGHTS)
    model = compile_model(model)
    print("Model loaded and compiled successfully.\n")
    
    # --- 2. Test Data Loading ---
    print("--- 2. Test Data Loading ---")
    # Validate test samples directory and load audio file paths
    if not os.path.isdir(TEST_SAMPLES_DIR):
        print(f"Error: Test samples directory '{TEST_SAMPLES_DIR}' not found.")
        return
    
    print(f"Loading test audio samples from '{TEST_SAMPLES_DIR}'...")
    test_audio_paths = load_wav_list_for_test(TEST_SAMPLES_DIR) 
    
    if not test_audio_paths:
        print(f"No .wav files found in '{TEST_SAMPLES_DIR}'.")
        return
    print(f"Found {len(test_audio_paths)} test audio files.\n")

    # --- 3. Data Preprocessing ---
    print("--- 3. Data Preprocessing ---")
    # X_lr_interpolated: Low-resolution audio, upsampled via interpolation to match model input.
    # Y_hr: Original high-resolution audio, used as the ground truth for evaluation.
    print("Preprocessing test audio samples into patches...")
    X_lr_interpolated, Y_hr, _ = preprocess( # Third return value (dataset_name_str) is not used here
        file_list=test_audio_paths, 
        start=0, 
        end=len(test_audio_paths)-1, 
        sr=SR, 
        scale=SCALE,
        dimension=DIMENSION,
        stride=STRIDE, 
        tag='test'  # Indicates test mode for preprocessing
    )

    if X_lr_interpolated is None or Y_hr is None or X_lr_interpolated.shape[0] == 0:
        print("Error: Preprocessing returned no data. Please check test files and parameters.")
        return

    print(f"Shape of X_lr_interpolated (model input patches): {X_lr_interpolated.shape}")
    print(f"Shape of Y_hr (ground truth HR patches): {Y_hr.shape}\n")

    # --- 4. Model Prediction ---
    print("--- 4. Model Prediction ---")
    print("Generating super-resolved audio with the model...")
    pred_hr = model.predict(X_lr_interpolated)
    print(f"Shape of pred_hr (model output patches): {pred_hr.shape}\n")

    # --- 5. Model Evaluation ---
    print("--- 5. Model Evaluation ---")
    print("Evaluating model performance against ground truth patches...")
    scores = model.evaluate(X_lr_interpolated, Y_hr, verbose=0) # verbose=0 to keep output clean
    print('Model Evaluation Scores:')
    for i, metric_name in enumerate(model.metrics_names):
        print(f'- {metric_name}: {scores[i]:.6f}') # Formatted for readability
    print("")

    # --- 6. Audio Preparation for STFT/Saving ---
    # Flatten audio data from patches to reconstruct continuous signals for STFT/saving
    y_hr_flat = Y_hr.flatten()
    x_lr_interpolated_flat = X_lr_interpolated.flatten()
    pred_hr_flat = pred_hr.flatten()

    # Generate a true low-resolution version from the first test sample for comparison
    # This avoids using the model's input (which is already interpolated) for "true LR" representation.
    true_lr_example_flat = np.array([])
    if test_audio_paths:
        print(f"Generating a true low-resolution example from: {test_audio_paths[0]}")
        first_sample_hr_audio, _ = librosa.load(test_audio_paths[0], sr=SR)
        true_lr_example_flat = first_sample_hr_audio[::SCALE]  # Direct downsampling

    # --- 7. STFT Plotting (Optional) ---
    # Ensure output directory for audio files exists (also used for plots if they save figs)
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)
    
    print(f"\n--- 7. STFT Plotting (matplotlib figures will pop up if backend is available) ---")
    print(f"Audio files for listening will be saved in '{OUTPUT_AUDIO_DIR}/'")
    # Generate and display STFT plots (if matplotlib is available and configured)
    STFT(y_hr_flat, title='Original High-Resolution (Ground Truth)', n_fft=2048, show=True)
    if true_lr_example_flat.size > 0:
        STFT(true_lr_example_flat, title=f'Original Low-Resolution (True {DOWNSAMPLED_SR}Hz quality - example)', n_fft=2048, show=True)
    STFT(x_lr_interpolated_flat, title='Low-Resolution Interpolated (Model Input)', n_fft=2048, show=True)
    STFT(pred_hr_flat, title='Super-Resolved (Model Output)', n_fft=2048, show=True)
    
    # --- 8. Saving Output Audio Files ---
    print(f"\n--- 8. Saving Output Audio Files to '{OUTPUT_AUDIO_DIR}/' ---")
    # 1. True Low-Resolution (example from the first test file)
    if true_lr_example_flat.size > 0:
        lr_path = os.path.join(OUTPUT_AUDIO_DIR, 'downsampled_true_lr_example.wav')
        sf.write(lr_path, true_lr_example_flat, DOWNSAMPLED_SR, 'PCM_24')
        print(f"Saved true low-resolution audio (example): {lr_path} (SR: {DOWNSAMPLED_SR})")
    
    # 2. Super-Resolved High-Resolution (model's output)
    sr_path = os.path.join(OUTPUT_AUDIO_DIR, 'superrezzed_output.wav')
    sf.write(sr_path, pred_hr_flat, SR, 'PCM_24')
    print(f"Saved super-resolved audio (model output): {sr_path} (SR: {SR})")

    # 3. Original High-Resolution (ground truth, for direct comparison)
    hr_path = os.path.join(OUTPUT_AUDIO_DIR, 'original_hr_ground_truth.wav')
    sf.write(hr_path, y_hr_flat, SR, 'PCM_24')
    print(f"Saved original high-resolution audio (ground truth): {hr_path} (SR: {SR})")

if __name__ == "__main__":
    print("Executing src/test.py as main script...")
    test_samples()
    print("\nTest script execution completed.")
