"""
src/models.py

Defines the neural network architectures used for audio super-resolution.
This includes the main U-Net like generator model (`base_model`) and
helper layers like `SubPixel1D` for upsampling.
"""
import keras
from keras.models import Sequential # Not directly used for base_model, but often imported with Keras
from keras.layers.core import Dense, Activation, Dropout # Activation and Dropout are used
from keras.layers.recurrent import LSTM, GRU # Not used in base_model
from keras.layers import Lambda, Conv1D # Lambda and Conv1D are used
from keras.layers.advanced_activations import LeakyReLU # Used
import tensorflow as tf

def split(x):
    """
    Slices a fixed range from the input tensor.

    This function was originally designed for a specific input/output patch configuration
    where the output (high-resolution center part) was extracted from a larger
    input patch. The range [:, 28:36] implies an input patch of size 64
    and an output patch (target) of size 8, taken from a specific offset.
    In the current `base_model`, this function is not directly used as the model
    predicts the full high-resolution patch.

    Args:
        x (tf.Tensor): Input tensor, expected to have at least 2 dimensions.
                       The slicing happens on the second dimension.

    Returns:
        tf.Tensor: The sliced tensor `x[:, 28:36]`.
    """
    return x[:,28:36] # it is fixed range for input(64) & output(8) dataset

def SubPixel1D(input_shape, r, color=False):
    """
    Implements a 1D subpixel convolution layer (also known as phase shift) for upsampling.

    Subpixel convolution reshapes the output of a convolutional layer with `r` times
    more channels into an output with `r` times larger spatial resolution by rearranging
    pixels from the channel dimension into the spatial dimension.

    Args:
        input_shape (tuple): Shape of the input tensor (batch_size, width, channels).
                             The number of channels must be divisible by `r`.
        r (int): The upsampling factor. The width of the output will be `width * r`.
        color (bool): Unused in this 1D context, kept for potential compatibility
                      or if adapted from a 2D version. Defaults to False.

    Returns:
        keras.layers.Lambda: A Keras Lambda layer that performs the subpixel upsampling.
    """
    def _phase_shift(I, r_factor=2):
        """
        Performs the phase shift operation to achieve subpixel upsampling.
        It transposes and reshapes the input tensor to move data from channels
        to spatial dimension.
        """
        X = tf.transpose(I, [2,1,0]) # (channels, width, batch_size)
        X = tf.batch_to_space_nd(X, [r_factor], [[0,0]]) # (channels/r, width*r, batch_size)
        X = tf.transpose(X, [2,1,0]) # (batch_size, width*r, channels/r)
        return X

    def subpixel_shape(input_shape_lambda):
        """
        Calculates the output shape of the subpixel layer.
        """
        dims = [input_shape_lambda[0],             # batch_size
            None if input_shape_lambda[1] is None else input_shape_lambda[1] * r, # width * r
            int(input_shape_lambda[2] / r)]      # channels / r
        output_shape = tuple(dims)
        return output_shape
  
    def subpixel(x_input):
        """
        Applies the phase shift operation.
        """
        # Note: The 'color' parameter is not used in this 1D implementation.
        # The comment "only single channel!" from the original notebook might be misleading,
        # as it works by rearranging multiple channels.
        x_upsampled = _phase_shift(x_input, r)
        return x_upsampled

    return Lambda(subpixel, output_shape=subpixel_shape, name=f'subpixel_upsample_x{r}')

def base_model(summary=True):
    """
    Defines and returns the base U-Net like model for audio super-resolution.

    The model architecture consists of:
    1. An encoder path (downsampling) with multiple 1D convolutional layers.
    2. A bottleneck layer.
    3. A decoder path (upsampling) with 1D convolutional layers and SubPixel1D layers for upsampling.
    4. Skip connections between corresponding layers in the encoder and decoder paths
       to help propagate features and improve gradient flow.
    5. A final residual connection adding the input to the output of the U-Net structure.

    Args:
        summary (bool, optional): If True, prints the model summary. Defaults to True.

    Returns:
        keras.models.Model: The compiled Keras model.
    """
    print('Loading base U-Net like model for audio super-resolution...')
    # Input layer: expects 1D sequences of length 256 with 1 channel (e.g., audio patches)
    main_input = keras.layers.Input((256,1), name='main_input')
    x = main_input
    
    # --- Encoder (Downsampling Path) ---
    # The comments like 'x1 = x # 128' indicate the feature map size after this block.
    
    # Downsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=16, kernel_size=16, activation=None, strides=2, name='enc_conv1')(x)
    x = LeakyReLU(0.2, name='enc_relu1')(x)
    x1 = x # Output size: 128
    
    # Downsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=8, activation=None, strides=2, name='enc_conv2')(x)
    x = LeakyReLU(0.2, name='enc_relu2')(x)
    x2 = x # Output size: 64
    
    # Downsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2, name='enc_conv3')(x)
    x = LeakyReLU(0.2, name='enc_relu3')(x)
    x3 = x # Output size: 32
    
    # Downsampling layer 4
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2, name='enc_conv4')(x)
    x = LeakyReLU(0.2, name='enc_relu4')(x)
    x4 = x # Output size: 16
    
    # Downsampling layer 5
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2, name='enc_conv5')(x)
    x = LeakyReLU(0.2, name='enc_relu5')(x)
    x5 = x # Output size: 8
    
    # Downsampling layer 6
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2, name='enc_conv6')(x)
    x = LeakyReLU(0.2, name='enc_relu6')(x)
    x6 = x # Output size: 4
    
    # --- Bottleneck Layer ---
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=4, activation=None, strides=2, name='bottleneck_conv')(x)
    x = LeakyReLU(0.2, name='bottleneck_relu')(x) # Output size: 2
    
    # --- Decoder (Upsampling Path) ---

    # Upsampling layer 6 (connects with x6 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None, name='dec_conv6a')(x) # Filters are 2*channels for SubPixel1D with r=2
    x = Activation('relu', name='dec_relu6a')(x)
    x = Dropout(rate=0.5, name='dec_drop6')(x)
    x = SubPixel1D(x.shape, r=2)(x) # Upsamples width by r=2, halves channels
    x = keras.layers.concatenate([x, x6], name='dec_concat6') # Skip connection
    
    # Upsampling layer 5 (connects with x5 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None, name='dec_conv5a')(x)
    x = Activation('relu', name='dec_relu5a')(x)
    x = Dropout(rate=0.5, name='dec_drop5')(x)
    x = SubPixel1D(x.shape, r=2)(x)
    x = keras.layers.concatenate([x, x5], name='dec_concat5') # Skip connection
     
    # Upsampling layer 4 (connects with x4 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None, name='dec_conv4a')(x)
    x = Activation('relu', name='dec_relu4a')(x)
    x = Dropout(rate=0.5, name='dec_drop4')(x)
    x = SubPixel1D(x.shape, r=2)(x)
    x = keras.layers.concatenate([x, x4], name='dec_concat4') # Skip connection
    
    # Upsampling layer 3 (connects with x3 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=4, activation=None, name='dec_conv3a')(x)
    x = Activation('relu', name='dec_relu3a')(x)
    x = Dropout(rate=0.5, name='dec_drop3')(x)
    x = SubPixel1D(x.shape, r=2)(x)
    x = keras.layers.concatenate([x, x3], name='dec_concat3') # Skip connection
    
    # Upsampling layer 2 (connects with x2 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*32, kernel_size=8, activation=None, name='dec_conv2a')(x)
    x = Activation('relu', name='dec_relu2a')(x)
    x = Dropout(rate=0.5, name='dec_drop2')(x)
    x = SubPixel1D(x.shape, r=2)(x)
    x = keras.layers.concatenate([x, x2], name='dec_concat2') # Skip connection
    
    # Upsampling layer 1 (connects with x1 from encoder)
    x = Conv1D(padding='same', kernel_initializer='Orthogonal',filters=2*16, kernel_size=16, activation=None, name='dec_conv1a')(x)
    x = Activation('relu', name='dec_relu1a')(x)
    x = Dropout(rate=0.5, name='dec_drop1')(x)
    x = SubPixel1D(x.shape, r=2)(x)
    x = keras.layers.concatenate([x, x1], name='dec_concat1') # Skip connection
   
    # --- Final Output Processing ---
    # Final convolution and SubPixel1D layer to bring to 1 channel and target resolution
    x = Conv1D(padding='same', kernel_initializer='he_normal',filters=2, kernel_size=8, activation=None, name='final_conv')(x) # filters=2 for SubPixel1D with r=2 to get 1 channel output
    x = SubPixel1D(x.shape, r=2)(x)
    
    # Residual connection: add the original input to the output of the U-Net
    output = keras.layers.add([x, main_input], name='residual_output')
    model  = keras.models.Model(main_input, output, name='audio_sr_unet')
    
    if summary: 
        model.summary()       
        
    return model

if __name__ == "__main__":
    # This part will only run when the script is executed directly.
    # It's useful for verifying the model architecture or performing standalone tests.
    print("Instantiating model for summary (if __name__ == '__main__'):")
    model_instance = base_model(summary=True)
    # Example: You could print input/output shapes here if needed
    # print(f"Model Input Shape: {model_instance.input_shape}")
    # print(f"Model Output Shape: {model_instance.output_shape}")
    print("models.py executed directly as main script.")
