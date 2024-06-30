# R2DNet Project Documentation

R2DNet (Reverberation to Decay Net) uses machine learning to offer an approach to estimating room acoustic decay parameters and noise floor estimation  from the reverberant speech of approximately 1 second. It is trained using energy decay curve loss for more roburst estimation in various acoustic environments.


## Directory Structure

This repository is organized into several sections, each dedicated to specific aspects of the acoustic analysis process:

### Data Preparation

- **IR_selection.py**: Selects impulse responses for testing and training datasets.
- **VAD.py**: Implements voice activity detection to identify speech segments.
- **dataset.py**: Manages loading of datasets for testing, validation, and room analysis.
- **reverb_preprocess_validation.py** & **reverbspeech_preprocess.py**: Generate reverberant speech segments for model input.

### Model Architecture and Training

- **model_FiNS.py**: Defines the FiNS model architecture.
- **model_alter.py**: Implements an encoder-decoder architecture for the S2IR model.
- **model_flex.py**: Offers a flexible CNN architecture for various dataset complexities.
- **s2ir.py**: Contains the architecture for the Speech-to-Impulse Response (S2IR) model.
- **training.py**: Facilitates model training with comprehensive configuration options.

### Testing and Analysis

- **test.py**: Executes model testing against predefined datasets.
- **process_utils.py**: Provides essential functions and loss calculations for model evaluation.
- **Analysis_plot_aggregate_analysis.py** & **Analysis_aggregate.py**: Perform detailed error and performance analysis.

### Utilities and Miscellaneous

- **hyperparameter_tuning.py**: Optimizes model hyperparameters using Optuna.
- **organise_test_data.py** & **organize_test_data_position.py**: Prepare test data, including positional information.

### Comprehensive MATLAB Script

- **speech_processing_individual_loc.m**: Consolidates reverberant speech data, labels, and other relevant information into single MATLAB files for each room/location, streamlining the dataset preparation process.

## Prerequisites

To work with the R2DNet framework, ensure you have Python 3.6+ and the following packages installed:


pip install numpy torch librosa optuna matplotlib seaborn

## Usage

Each script can be executed individually to perform its designated function. For example, to generate reverberant speech data for validation:


python reverb_preprocess_validation.py
