"""
src/utils.py

This script contains various utility functions for the audio super-resolution project.
These include:
- Custom Keras loss/metric functions (SNR, sum_loss).
- Model loading and compilation helpers.
- Signal processing routines such as upsampling, Short-Time Fourier Transform (STFT),
  Inverse STFT (iSTFT), and functions for plotting waveforms and spectrums.
- Data conversion utilities (e.g., float to complex).
"""
import librosa
import numpy as np
from scipy import interpolate
from scipy.signal import decimate # decimate is imported but not used in this file
from matplotlib import pyplot as plt
from scipy import signal # signal is imported but not used directly, though submodules like decimate are
import h5py # h5py is imported but not used in this file
import wave # wave is imported but not used in this file
from keras import backend as K

def SNR(y_true, y_pred):
    """
    Calculates the Signal-to-Noise Ratio (SNR) as a Keras metric.

    SNR is a measure used to compare the level of a desired signal to the
    level of background noise. It is defined as the ratio of signal power to
    the noise power, often expressed in decibels (dB).

    Args:
        y_true (tf.Tensor): The ground truth signal.
        y_pred (tf.Tensor): The predicted signal.

    Returns:
        tf.Tensor: A scalar tensor representing the average SNR in dB.
    """
    P = y_pred # Predicted signal
    Y = y_true # True signal
    
    # Calculate L2 loss (mean squared error) between prediction and true signal
    # Adding a small epsilon (1e-6) for numerical stability.
    sqrt_l2_loss = K.sqrt(K.mean((P - Y)**2 + 1e-6))
    
    # Calculate L2 norm (root mean square) of the true signal
    sqrn_l2_norm = K.sqrt(K.mean(Y**2))
    
    # Calculate SNR in dB. Adding epsilon (1e-8) to avoid log(0).
    snr = 20 * K.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / K.log(10.)
    
    avg_snr = K.mean(snr)
    return avg_snr

def sum_loss(y_true, y_pred):
    """
    Calculates the sum of squared errors loss.

    This is a simple loss function that computes the sum of the squared
    differences between the true and predicted values.

    Args:
        y_true (tf.Tensor): The ground truth values.
        y_pred (tf.Tensor): The predicted values.

    Returns:
        tf.Tensor: A scalar tensor representing the sum of squared errors.
    """
    P = y_pred
    Y = y_true
    loss = K.sum((P - Y)**2)
    return loss

def load_model(model, weights_file, load_weights=False):
    """
    Loads model weights from a file if specified.

    Args:
        model (keras.models.Model): The Keras model instance.
        weights_file (str): Path to the HDF5 file containing the model weights.
        load_weights (bool, optional): If True, attempts to load the weights
                                       into the model. Defaults to False.

    Returns:
        keras.models.Model: The model, with weights loaded if requested and successful.
    """
    if load_weights: 
        try:
            model.load_weights(weights_file)
            print(f'Successfully loaded model weights from {weights_file}')
        except Exception as e:
            print(f"Error loading weights from {weights_file}: {e}. Starting with an untrained model.")
    return model

def compile_model(model):
    """
    Compiles a Keras model with Mean Squared Error (MSE) loss, Adam optimizer,
    and custom metrics (sum_loss, SNR).

    Args:
        model (keras.models.Model): The Keras model to compile.

    Returns:
        keras.models.Model: The compiled Keras model.
    """
    model.compile(loss='mse', optimizer="adam", metrics=[sum_loss, SNR])
    print('Model compilation successful with MSE loss, Adam optimizer, sum_loss and SNR metrics.')
    return model

def wav2plot(signal_bytes_data):
    """
    Plots a signal wave from raw audio bytes.

    The input is expected to be a byte string representing audio data,
    which is then converted to 'Int16' numpy array for plotting.

    Args:
        signal_bytes_data (bytes): Raw audio data as a byte string.
                                   Example: `wav_file.readframes(-1)`
    """
    # Convert raw audio bytes (assuming 'Int16' format) to numpy array
    signal_np_array = np.frombuffer(signal_bytes_data, dtype=np.int16) # Use frombuffer for bytes
    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(signal_np_array)
    plt.show()

def float2complex(float_data_array):
    """
    Converts a float array with real and imaginary parts stacked along the
    last dimension into a complex numpy array.

    Args:
        float_data_array (np.ndarray): A numpy array where the last dimension
                                       has size 2, representing [real, imag].
                                       Shape: (..., 2)

    Returns:
        np.ndarray: A complex numpy array. Shape: (...)
    """
    # fdata[...,0] is the real part, fdata[...,1] is the imaginary part
    complex_array = float_data_array[...,0] + 1j * float_data_array[...,1]
    return complex_array

def upsample(low_res_signal, upsampling_factor):
    """
    Upsamples a 1D low-resolution signal using cubic spline interpolation.

    Args:
        low_res_signal (np.ndarray): The 1D low-resolution input signal (flattened).
        upsampling_factor (int): The factor by which to upsample the signal.

    Returns:
        np.ndarray: The upsampled 1D signal.
    """
    low_res_signal = low_res_signal.flatten()
    low_res_length = len(low_res_signal)
    high_res_length = low_res_length * upsampling_factor
    
    # Create an array of x-coordinates for the low-resolution signal points
    original_x_coords = np.arange(0, high_res_length, upsampling_factor)
    # Create an array of x-coordinates for the desired high-resolution signal points
    new_x_coords = np.arange(high_res_length)

    # Perform cubic spline interpolation
    # splrep finds the B-spline representation of a 1-D curve.
    spline_representation = interpolate.splrep(original_x_coords, low_res_signal)
    # splev evaluates a B-spline or its derivatives.
    high_res_signal = interpolate.splev(new_x_coords, spline_representation)

    return high_res_signal

def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Custom padding function for `np.pad`. Pads with a specified value.
    This function signature is designed to be compatible with `np.pad`'s `func` argument.

    Args:
        vector (np.ndarray): The vector to be padded.
        pad_width (tuple): A tuple of (before_val, after_val) indicating the number of values
                           padded to the edges of each axis.
        iaxis (int): The axis currently being padded.
        kwargs (dict): A dictionary of keyword arguments. Expects 'padder' key
                       for the value to pad with. Defaults to 0 if 'padder' not found.

    Returns:
        np.ndarray: The padded vector.
    """
    pad_value = kwargs.get('padder', 0)
    # This function is called for each axis. pad_width[0] is padding at the beginning,
    # pad_width[1] is padding at the end. This implementation only pads the beginning.
    # A more general implementation would use both:
    # vector[:pad_width[0]] = pad_value
    # vector[-pad_width[1]:] = pad_value 
    vector[:pad_width[0]] = pad_value # Example: pads only the beginning
    # If padding both ends is needed, the logic should be adjusted.
    # For instance, vector[pad_width[0]:-pad_width[1]] would be the original data.
    return vector

def STFT(audio_signal, title, n_fft=2048, show=False):
    """
    Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    The STFT represents a signal in the time-frequency domain by computing
    discrete Fourier transforms (DFT) over short overlapping windows.

    Args:
        audio_signal (np.ndarray): The 1D input audio signal.
        title (str): Title for the spectrum plot if `show` is True.
        n_fft (int, optional): Length of the FFT window. Defaults to 2048.
        show (bool, optional): If True, displays the spectrum using `showSpectrum`.
                               Defaults to False.

    Returns:
        tuple: A tuple containing:
            - S_data (np.ndarray): STFT complex results concatenated along the last axis
                                   as [real_part, imaginary_part]. Shape: (num_frames, num_bins, 2)
            - S_complex (np.ndarray): Complex-valued STFT matrix. Shape: (num_bins, num_frames)
                                      Note: librosa.stft returns (num_bins, num_frames). This function transposes it.
    """
    print(f"Input signal for STFT (shape: {audio_signal.shape}, first few values: {audio_signal[:5]})")
    # librosa.stft returns a complex matrix D where:
    # D[f, t] is the complex value for frequency bin f at frame t.
    # Shape is (1 + n_fft/2, n_frames)
    S_complex_librosa = librosa.stft(audio_signal, n_fft=n_fft)
    
    # Transpose to have frames as the first dimension, frequency bins as the second. (num_frames, num_bins)
    S_complex_transposed = S_complex_librosa.T 
    
    # Expand dimensions to separate real and imaginary parts for potential model input
    S_real = np.expand_dims(S_complex_transposed.real, axis=-1) 
    S_imag = np.expand_dims(S_complex_transposed.imag, axis=-1)
    # Concatenate real and imaginary parts to form a (num_frames, num_bins, 2) array
    S_data_stacked = np.concatenate((S_real, S_imag), axis=-1)
    
    if show:
        # showSpectrum expects the original librosa output shape (num_bins, num_frames)
        # or its transpose if imshow is to display time along x-axis.
        # The current showSpectrum transposes its input, so pass S_complex_transposed.
        showSpectrum(S_complex_transposed, title) # S_complex_transposed is (num_frames, num_bins)
                                             # showSpectrum will then plot it as (num_bins, num_frames)
        
    return S_data_stacked, S_complex_transposed # Return the transposed complex STFT

def showSpectrum(S_complex_matrix, title, aspect_ratio=0.1):
    """
    Displays the magnitude spectrum of an STFT result.

    Args:
        S_complex_matrix (np.ndarray): Complex-valued STFT matrix.
                                       Expected shape (num_frames, num_bins) or (num_bins, num_frames).
                                       The function transposes it for `plt.imshow`.
        title (str): Title for the plot.
        aspect_ratio (float, optional): Aspect ratio for the imshow plot. Defaults to 0.1.
    """
    # S_complex_matrix could be (num_frames, num_bins) from the modified STFT function
    # or (num_bins, num_frames) if directly from librosa.
    # imshow(X.T) means if X is (frames, bins), it plots (bins, frames) i.e. bins on y, frames on x.
    # If X is (bins, frames), it plots (frames, bins) i.e. frames on y, bins on x.
    # The current STFT passes (num_frames, num_bins). So S_complex_matrix.T is (num_bins, num_frames).
    # This means frequency bins on y-axis, time frames on x-axis.
    
    abs_S = np.abs(S_complex_matrix)
    # Use log scale for better visualization (log1p handles zeros by adding 1 before log)
    log_magnitude_S = np.log1p(abs_S)
    
    plt.figure(figsize=(20,10))
    # plt.imshow expects (rows, columns), so (frequency_bins, time_frames)
    # If S_complex_matrix is (num_frames, num_bins), its transpose is (num_bins, num_frames)
    plt.imshow(log_magnitude_S.T, aspect=aspect_ratio, origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB (approx)') # Add a colorbar
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.show()

def iSTFT(S_complex_matrix_transposed):
    """
    Computes the Inverse Short-Time Fourier Transform (iSTFT) of a complex STFT matrix.

    Args:
        S_complex_matrix_transposed (np.ndarray): Complex-valued STFT matrix,
                                                  expected shape (num_frames, num_bins).
                                                  This will be transposed back before librosa.istft.

    Returns:
        np.ndarray: The reconstructed 1D time-domain audio signal.
    """
    # librosa.istft expects input shape (num_bins, num_frames)
    S_complex_for_librosa = S_complex_matrix_transposed.T
    reconstructed_signal = librosa.istft(S_complex_for_librosa)
    return reconstructed_signal

# Adding an if __name__ == "__main__": block for any potential test/demonstration code
if __name__ == "__main__":
    # This block is for testing utility functions if the script is run directly.
    print("utils.py executed directly. Demonstrating some utility functions:")

    # Example for upsample
    print("\n--- Upsample Demo ---")
    lr_signal_test = np.array([1, 2, 3, 4])
    upsampled_signal_test = upsample(lr_signal_test, 2)
    print(f"Original LR signal: {lr_signal_test}")
    print(f"Upsampled (x2) signal: {upsampled_signal_test}")

    # Example for STFT and iSTFT
    print("\n--- STFT/iSTFT Demo ---")
    # Create a simple test sine wave: 1 second, 440 Hz tone at 8000 Hz sample rate
    sample_rate_stft = 8000
    duration_stft = 1
    frequency_stft = 440
    t_stft = np.linspace(0, duration_stft, int(sample_rate_stft * duration_stft), endpoint=False)
    test_audio_stft = 0.5 * np.sin(2 * np.pi * frequency_stft * t_stft)
    
    print(f"Created a test audio signal: {len(test_audio_stft)} samples, {duration_stft}s, {frequency_stft}Hz tone at {sample_rate_stft}Hz SR.")
    
    # Perform STFT
    # Note: show=True will attempt to plot, which might require a GUI backend.
    # Set show=False if running in a non-GUI environment for automated tests.
    S_data_test, S_complex_test = STFT(test_audio_stft, "Test STFT Spectrum", n_fft=512, show=False) 
    print(f"STFT data shape (real/imag stacked): {S_data_test.shape}")
    print(f"STFT complex matrix shape (frames, bins): {S_complex_test.shape}")

    # Perform iSTFT
    restored_signal_test = iSTFT(S_complex_test)
    print(f"Restored signal shape: {restored_signal_test.shape}")

    # Check if restored signal is close to original (within numerical precision limits)
    # Note: Length might differ slightly due to STFT padding/framing.
    min_len = min(len(test_audio_stft), len(restored_signal_test))
    diff_stft = np.abs(test_audio_stft[:min_len] - restored_signal_test[:min_len])
    print(f"Mean absolute difference between original and restored signal (first {min_len} samples): {np.mean(diff_stft):.2e}")

    # Example for wav2plot (requires a byte string)
    # print("\n--- wav2plot Demo (plotting disabled in this example) ---")
    # # Create a dummy wave file in memory to get bytes
    # import io
    # dummy_wav_file = io.BytesIO()
    # with wave.open(dummy_wav_file, 'wb') as wf_test:
    #     wf_test.setnchannels(1)
    #     wf_test.setsampwidth(2) # 16-bit
    #     wf_test.setframerate(8000)
    #     wf_test.writeframes((test_audio_stft * 32767).astype(np.int16).tobytes()) # Scale to int16
    # signal_bytes_for_plot = dummy_wav_file.getvalue() # This is actually the full WAV file bytes
    # # To get raw frames:
    # # signal_bytes_for_plot = (test_audio_stft * 32767).astype(np.int16).tobytes()
    # # wav2plot(signal_bytes_for_plot) # This would show a plot
    # print("wav2plot would attempt to plot the signal if matplotlib backend is available.")
    print("\n--- End of Demos ---")
