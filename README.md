# Audio Super Resolution for Improved Speech Recognition
A project to use audio super resolution to upsample english speech and feed to speech recognition softwares for better accuracy

The paper for audio super resolution: https://arxiv.org/pdf/1708.00853.pdf :AUDIO SUPER-RESOLUTION USING NEURAL NETS by Volodymyr Kuleshov, S. Zayd Enam, and Stefano Ermon

The code has been adapted from the following github repo:https://github.com/leekh7411/Fast-Audio-Super-Resolution
The network has been trained on the VCTK corpus as provided by University of Edinburgh: https://datashare.is.ed.ac.uk/handle/10283/2651

Major changes made to the repo are as follows:
- All code has been ported to notebooks for easier runtime accessibility.
- Dataset loading has been modified for the current structure of VCTK corpus.
- Final predictions have been converted to audio files with the intended sampling rates for qualitative and quantitave tests on aforementioned audio files.
- Changed training hyperparameters for slightly improved accuracies at the cost of a slightly longer training time (it is still incredibly fast on any mid to high tier gpu)
- Other minor bug fixes.
- Converted Jupyter notebooks to Python scripts located in the `src/` directory.
- Centralized path configurations in `src/config.py`.

## Setup

### Dependencies

The project requires the following Python dependencies:

```
keras==2.3.1
tensorflow==1.15.2
librosa==0.9.2
numpy==1.19.5
scipy==1.7.3
matplotlib==3.4.3
h5py==2.10.0
```

You can install these dependencies using pip and the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Configuration

Before running the scripts, you may need to configure paths in `src/config.py`. 
Specifically, ensure `VCTK_CORPUS_PATH` points to the root directory of your VCTK (or other speech) corpus. 
Other paths for preprocessed data, weights, and test outputs are also defined in this file and will be created if they don't exist by the relevant scripts.

## Order of Execution

The scripts are designed to be run in the following order:

1.  **`src/models.py`**: Defines the neural network model. This script is typically imported by `train.py` and `test.py`, not run directly unless you want to inspect the model structure.
2.  **`src/utils.py`**: Contains utility functions used by other scripts. Also typically imported.
3.  **`src/prep_data.py`**: Preprocesses the raw audio data from the corpus specified in `src/config.py` (e.g., VCTK_CORPUS_PATH) and saves it into HDF5 files in the `PREPROCESSED_DATA_PATH` (also defined in `src/config.py`). Run this script to prepare your dataset.
    ```bash
    python src/prep_data.py
    ```
4.  **`src/train.py`**: Trains the model using the preprocessed data. Model weights will be saved to the directory specified in `WEIGHTS_PATH` in `src/config.py`.
    ```bash
    python src/train.py
    ```
5.  **`src/test.py`**: Tests the trained model using sample audio files from `TEST_SAMPLES_PATH` (configurable in `src/config.py`). Output audio files will be saved in `OUTPUT_AUDIO_PATH`.
    ```bash
    python src/test.py
    ```

The scripts `Speech_Recognition/comparer.py` and `Speech_Recognition/speech_evaluation.py` are utility scripts for comparing text files and can be run independently. They now accept command-line arguments for input files. For example:
```bash
python Speech_Recognition/comparer.py "your query string" path/to/database_file.txt
python Speech_Recognition/speech_evaluation.py path/to/fileA.txt path/to/fileB.txt
```
