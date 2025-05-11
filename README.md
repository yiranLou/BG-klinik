# OpenSim Pipeline for Upper Limb Biomechanics

## Project Overview

This project implements a complete OpenSim data processing pipeline, focusing on upper limb biomechanics analysis, particularly EMG and kinematics analysis of wrist and finger muscles. The project integrates the entire process of data collection, processing, analysis, and visualization, combined with muscle synergy theory and neural network methods.

## Features

- **C3D Data Conversion**: Convert motion capture data to OpenSim compatible format
- **EMG Signal Processing**: Wavelet transform preprocessing and MVIC normalization
- **OpenSim Core Functions**: Model scaling, inverse kinematics, inverse dynamics, forward dynamics
- **Muscle Synergy Analysis**: Muscle synergy extraction based on non-negative matrix factorization (NMF)
- **Neural Network Integration**: Training neural networks to map muscle synergies to joint dynamics
- **Spasticity Analysis**: Analysis of EMG differences during slow and fast passive stretching

## Project Structure

```
opensim_pipeline/
│
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── opensim_config.yaml          # OpenSim configuration parameters
│   ├── emg_config.yaml              # EMG processing parameters  
│   ├── models/                      # OpenSim model files
│   │   ├── upper_limb_hand.osim     # Upper limb and hand model
│   │   └── markers.xml              # Marker definitions
│   └── scaling_config.yaml          # Scaling parameters
│
├── data/                            # Data storage
│   ├── raw/                         # Raw experimental data
│   │   ├── emg/                     # Raw EMG data
│   │   ├── motion_capture/          # Raw motion capture data (C3D)
│   │   └── mvic/                    # Maximum voluntary isometric contraction data
│   ├── processed/                   # Processed data
│   │   ├── emg/                     # Processed EMG
│   │   ├── kinematics/              # Processed kinematics data
│   │   └── synergies/               # Extracted muscle synergies
│   └── results/                     # Analysis results
│       ├── figures/                 # Generated figures
│       └── tables/                  # Generated tables
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── preprocessing/               # Data preprocessing
│   │   ├── __init__.py
│   │   ├── c3d_converter.py         # convert to trc and mot
│   │   ├── emg_processing.py        # EMG signal processing
│   │   ├── wavelet_transform.py     # EMG wavelet transform
│   │   ├── normalization.py         # Data normalization (MVIC)
│   │   └── kinematics_processing.py # Kinematics data processing
│   │
│   ├── opensim/                     # OpenSim wrapper
│   │   ├── __init__.py
│   │   ├── scaling.py               # Model scaling tools
│   │   ├── inverse_kinematics.py    # Inverse kinematics
│   │   ├── inverse_dynamics.py      # Inverse dynamics
│   │   ├── forward_dynamics.py      # Forward dynamics
│   │   └── visualization.py         # OpenSim visualization
│   │
│   ├── synergy/                     # Muscle synergy analysis
│   │   ├── __init__.py
│   │   ├── nmf.py                   # Non-negative matrix factorization
│   │   ├── synergy_extraction.py    # Muscle synergy extraction
│   │   └── synergy_analysis.py      # Synergy analysis tools
│   │
│   ├── neural_network/              # Neural network models
│   │   ├── __init__.py
│   │   ├── models.py                # Neural network model definitions
│   │   ├── training.py              # Model training tools
│   │   └── evaluation.py            # Model evaluation
│   │
│   ├── analysis/                    # Data analysis
│   │   ├── __init__.py
│   │   ├── spasticity_analysis.py   # Spasticity analysis (EMG_HV - EMG_LV)
│   │   ├── range_of_motion.py       # Range of motion analysis
│   │   └── statistical_analysis.py  # Statistical analysis
│   │
│   └── visualization/               # Visualization tools
│       ├── __init__.py
│       ├── plot_emg.py              # EMG plotting
│       ├── plot_kinematics.py       # Kinematics plotting
│       └── plot_synergies.py        # Synergy plotting
│
├── scripts/                         # Executable scripts
│   ├── process_c3d.py               # Process C3D files
│   ├── preprocess_emg.py            # Preprocess EMG data
│   ├── extract_synergies.py         # Extract muscle synergies
│   ├── scale_model.py               # Scale OpenSim model
│   ├── run_ik.py                    # Run inverse kinematics
│   ├── run_id.py                    # Run inverse dynamics
│   ├── train_nn.py                  # Train neural network
│   └── run_fd.py                    # Run forward dynamics
│
├── notebooks/                       # Jupyter notebooks
│   ├── emg_preprocessing.ipynb      # EMG preprocessing
│   ├── kinematic_analysis.ipynb     # Kinematic analysis
│   ├── synergy_extraction.ipynb     # Synergy extraction analysis
│   └── neural_network_training.ipynb # Neural network training
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_c3d_converter.py        # Test C3D conversion
│   ├── test_emg_processing.py       # Test EMG processing
│   ├── test_scaling.py              # Test model scaling
│   └── test_nmf.py                  # Test NMF implementation
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation script
└── README.md                        # Project documentation
```

## Installation Guide

### Prerequisites

- Python 3.8+
- OpenSim 4.5+
- Numpy, Scipy, Pandas, Matplotlib
- PyTorch or TensorFlow
- PyWavelets

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/opensim_pipeline.git
cd opensim_pipeline
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install OpenSim Python API
```bash
# Install the appropriate version according to your operating system following the official OpenSim guide
```

5. Install this project package
```bash
pip install -e .
```

## Experimental Methods and Parameters

### Surface Electromyography (sEMG)

#### Measured Muscle Groups
- **Wrist Flexors**: Flexor Carpi Radialis (FCR), Flexor Carpi Ulnaris (FCU)
- **Wrist Extensors**: Extensor Carpi Radialis Longus (ECRL), Extensor Carpi Radialis Brevis (ECRB), Extensor Carpi Ulnaris (ECU)
- **Finger (including thumb) Long Flexors and Extensors**:
  - Flexor Digitorum Superficialis (FDS), Flexor Digitorum Profundus (FDP), Extensor Digitorum Communis (EDC)
  - Flexor Pollicis Longus (FPL) and Extensor Pollicis Longus (EPL)

#### Placement Guidelines
- Follow SENIAM guidelines for electrode placement and procedures

#### Normalization
- Use Maximum Voluntary Isometric Contraction (MVIC) tests
- Convert raw EMG values to %MVIC for comparison

#### Avoiding Active Participation
- Monitor antagonist muscle activity to exclude interference from active effort

### 3D Motion Capture

#### Basic Principles
- Use infrared reflective markers
- Multiple cameras simultaneously capture changes in marker spatial positions

#### Adopted Models
- **U.L.E.M.A. model**: For large upper limb joints (shoulder, elbow, wrist)
- **HAWK 26-marker-set**: For three-dimensional kinematic information of finger joints

#### Virtual Markers and Pointer Calibration Techniques
- Use "clusters" to reduce errors from skin movement
- More precise identification of joint centers and motion axes

#### Measurement Parameters
- Joint angles (e.g., MCP, PIP, DIP, wrist joint)
- Angular velocity (first derivative)

### Experimental Procedure

#### Passive Stretching (slow vs. fast)
- **Slow**: Target 30°/s
- **Fast**: Target 180°/s
- 3 repetitions for each speed, with 5-second rest intervals

#### Measurement Sequence
- Passive wrist extension under different conditions
- Individual stretching of each finger or thumb

### Key Output Metrics

#### EMG Parameters
- **EMG_LV**: Average EMG during slow passive stretching (%MVIC)
- **EMG_HV**: Average EMG during fast stretching
- **EMG_change = EMG_HV - EMG_LV**: Reflects spasticity

#### Maximum Extension Limitation Angle
- Maximum extension range achievable for each joint

## Usage Guide

### Data Preprocessing

1. Process C3D files
```bash
python scripts/process_c3d.py --input data/raw/motion_capture --output data/processed/kinematics
```

2. Preprocess EMG data
```bash
python scripts/preprocess_emg.py --input data/raw/emg --output data/processed/emg --config config/emg_config.yaml
```

3. Extract muscle synergies
```bash
python scripts/extract_synergies.py --input data/processed/emg --output data/processed/synergies --num-synergies 4
```

### OpenSim Workflow

1. Scale model
```bash
python scripts/scale_model.py --model config/models/upper_limb_hand.osim --markers data/processed/kinematics/static_trial.trc --output models/scaled_model.osim
```

2. Run inverse kinematics
```bash
python scripts/run_ik.py --model models/scaled_model.osim --markers data/processed/kinematics/dynamic_trial.trc --output data/processed/kinematics/joint_angles.mot
```

3. Train neural network
```bash
python scripts/train_nn.py --synergies data/processed/synergies --kinematics data/processed/kinematics/joint_angles.mot --output models/nn_model.pt
```

4. Run forward dynamics
```bash
python scripts/run_fd.py --model models/scaled_model.osim --nn-model models/nn_model.pt --synergies data/processed/synergies/test_data.csv --output data/results/fd_results.mot
```

### Analysis and Visualization

1. Spasticity analysis
```bash
python scripts/analyze_spasticity.py --emg-slow data/processed/emg/slow_stretch.csv --emg-fast data/processed/emg/fast_stretch.csv --output data/results/spasticity_analysis.csv
```

2. Use Jupyter notebooks for interactive analysis
```bash
jupyter notebook notebooks/
```

## Technical Implementation Highlights

### 1. Wavelet Transform Preprocessing

- Select appropriate wavelet basis function (Daubechies db4)
- 3-5 level wavelet decomposition
- Threshold denoising
- Implementation using PyWavelets library

### 2. Muscle Synergy Extraction (NMF)

- Use sklearn.decomposition.NMF or nimfa library
- EMG ≈ W × H matrix factorization
- Select appropriate number of synergies R (2-4)

### 3. Neural Network Mapping

- Input: Muscle synergy activation coefficients h(t)
- Output: Joint angles, angular velocities, or muscle activation values
- Network scale: 2-4 layer MLP, 32-128 neurons/layer
- Avoid overfitting: Data augmentation, cross-validation

### 4. OpenSim Integration

- Offline prediction + OpenSim driving
- Create custom Controller through Python API
- Evaluation: RMSE, correlation coefficient, DTW error

## Developer Guide

### Code Standards
- Follow PEP 8
- Use type annotations
- Write documentation strings for all functions

### Testing
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_emg_processing.py
```

### Contribution Guidelines
1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Author

- Your Name - [Your Email]

# OpenSim Model Scaling Tool

This directory contains scripts to scale an OpenSim model and save the results to a specified location.

## Files

- `scale_model.py`: The main Python script that uses the Scale class to scale an OpenSim model.
- `run_scale_model.bat`: A Windows batch file to simplify running the Python script with parameters.

## Requirements

- Python 3.6+
- OpenSim Python API
- The Scale class from the opensimarmmodel-main module

## Usage

### Using the batch file

1. Edit the `run_scale_model.bat` file to set the correct paths and parameters:
   - `MODEL_INPUT`: Path to your generic OpenSim model (.osim file)
   - `XML_INPUT`: Path to your generic scaling XML configuration
   - `STATIC_PATH`: Path to your static trial (.trc file)
   - `MASS`: Participant's mass in kg
   - `HEIGHT`: Participant's height in mm
   - `AGE`: Participant's age in years
   - `OUTPUT_NAME`: Base name for the output files

2. Run the batch file by double-clicking it or from the command prompt.

### Using the Python script directly

You can also run the Python script directly with the required parameters:

```
python scale_model.py --model_input <path_to_model> --xml_input <path_to_xml> --static_path <path_to_trc> --mass <weight_in_kg> [--height <height_in_mm>] [--age <age_in_years>] [--add_model <path_to_additional_model>] [--remove_unused] [--output_name <base_name>]
```

### Required Parameters

- `--model_input`: Path to the generic model (.osim file)
- `--xml_input`: Path to the generic scaling XML configuration
- `--static_path`: Path to the static trial (.trc file)
- `--mass`: Participant's mass in kg

### Optional Parameters

- `--height`: Participant's height in mm (default: -1)
- `--age`: Participant's age in years (default: -1)
- `--add_model`: Path to a model to append
- `--remove_unused`: Flag to remove unused markers
- `--output_name`: Base name for output files (default: 'scaled_model')

## Output

All output files will be saved to:
```
C:/temporary_file/BG_klinik/newPipeline/data/processed
```

The script will generate:
- A scaled OpenSim model (.osim file)
- A scaling configuration XML file
- Potentially other files based on the Scale class functionality

## Troubleshooting

If you encounter errors:

1. Make sure Python and OpenSim are properly installed
2. Check that all paths to input files are correct
3. Verify that the Scale class is accessible in your Python environment
4. Ensure the output directory exists or can be created

# Neural Network Pipeline for Synergy-to-Motion Mapping

This project implements a comprehensive neural network pipeline for mapping muscle synergy activation patterns to joint kinematics. It includes multiple neural network architectures, advanced training techniques, data augmentation, and detailed evaluation metrics.

## Features

- **Multiple Neural Network Models**: Transformer, Attention LSTM, ConvLSTM, Residual LSTM, and Ensemble models
- **Advanced Training Techniques**: Gradient clipping, learning rate warmup, mixed precision training
- **Data Augmentation**: Noise injection, time warping, scaling, mixup, and cutmix
- **Comprehensive Evaluation**: RMSE, MAE, R-squared, uncertainty estimation, and detailed joint-wise metrics
- **Visualization**: Training curves, prediction plots, error distributions, and residual analysis

## Project Structure

```
newPipeline/
├── config/              # Configuration files
├── data/                # Data directory
│   ├── processed/       # Processed data
│   └── results/         # Results and model outputs
├── scripts/             # Script files
│   └── run_neural_network_comparison.py  # Main training script
└── src/                 # Source code
    └── neural_network/  # Neural network modules
        ├── data_loader.py           # Data loading utilities
        ├── models.py                # Model architectures
        ├── trainer.py               # Training utilities
        ├── evaluator.py             # Evaluation utilities
        └── data_augmentation.py     # Data augmentation techniques
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/newPipeline.git
cd newPipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default settings
python scripts/run_neural_network_comparison.py

# Run with custom settings
python scripts/run_neural_network_comparison.py --models_to_compare attention_lstm transformer --sequence_length 10 --hidden_dim 64 --epochs 100
```

## Configuration Options

- `--synergy_path`: Path to synergy activation patterns
- `--ik_path`: Path to IK results
- `--output_path`: Path to save results
- `--models_to_compare`: List of models to compare
- `--sequence_length`: Sequence length for models
- `--hidden_dim`: Hidden dimension for models
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--use_augmentation`: Use data augmentation
- `--use_enhanced_features`: Use enhanced feature extraction

## License

[MIT License](LICENSE)