# Pump-Probe Spectroscopy Analysis

A comprehensive Python package for analyzing pump-probe spectroscopy imaging data, with a focus on time-resolved transient absorption microscopy.

## Features

- **Multiple Data Format Support**: Import from DukeScan, Mathematica, and pickle formats
- **Data Processing**: Background subtraction, normalization, spatial masking, and downsampling
- **Phasor Analysis**: Frequency-domain visualization for identifying decay patterns
- **Machine Learning Classification**: Pixel-wise classification of different material types
- **Intensity Thresholding**: Automated and manual masking based on signal intensity
- **Visualization Tools**: Interactive plotting of TA curves, projections, and phasor plots
- **Linear Combinations**: Arithmetic operations on multiple stacks

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd pump_probe_analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from pps import PPS
from pathlib import Path

# Load a pump-probe stack from DukeScan format
filename = Path("data/example_stack_DS_CH1.tif")
stack = PPS(filename, dataType="DukeScan")

# Subtract background (average of first 3 frames)
stack.subtractFirst(n=3)

# Normalize the data
stack.normalize(norm="minmax")

# View average transient absorption curve
stack.avg_show()

# View projection (sum of absolute values)
stack.project_show()
```

### Phasor Analysis

```python
# Perform phasor analysis at 0.25 THz
phasor_data = stack.phasor(freq=0.25, remove_zero=True)

# Visualize phasor plot
stack.phasor_show(freq=0.25)
```

### Intensity Thresholding

```python
# Apply automatic Li threshold
stack.intensity_threshold(threshold="Li", sigma=5)

# Or use manual threshold
stack.intensity_threshold(threshold=0.1, sigma=5)

# View the mask
stack.mask_show()
```

### Time Delay Selection

```python
# Select specific time delays
delays = [-1.0, 0.0, 0.5, 1.0, 5.0, 10.0, 50.0]
stack.select_delays(delays=delays)

# Or use predefined melanoma preset
stack.select_delays(delays="melanoma1")
```

### Downsampling

```python
# Downsample by factor of 2 to improve SNR
stack_ds = stack.downsample(size=2)
```

### Classification

```python
from sklearn.ensemble import RandomForestClassifier

# Train your classifier (example)
# classifier = train_classifier()  # Your training code

# Classify pixels
stack.classify_show(classifier, downsample=2, norm="minmax")
```

### Linear Combinations

```python
# Subtract two stacks
difference = PPS.linear_combination(stack1, 1, stack2, -1)

# Average two stacks
average = PPS.linear_combination(stack1, 0.5, stack2, 0.5)
```

## Module Structure

### `pps.py`
Main module containing the `PPS` class for pump-probe stack analysis.

**Key Methods:**
- `__init__()`: Import data from various formats
- `save()`: Save stack to pickle format
- `subtractFirst()`: Background subtraction
- `normalize()`: Data normalization
- `avg()`, `avg_show()`: Average TA curves
- `project()`, `project_show()`: Stack projections
- `phasor()`, `phasor_show()`: Phasor analysis
- `intensity_threshold()`: Automatic masking
- `classify()`, `classify_show()`: ML classification
- `downsample()`: Spatial downsampling
- `substacks()`: Divide into spatial regions
- `linear_combination()`: Arithmetic on stacks

### `ta.py`
Transient absorption model functions for fitting decay dynamics.

**Functions:**
- `decay_single()`: Single exponential decay model
- `decay_infinite()`: Infinite lifetime model

### `fit.py`
Functions for fitting experimental data.

**Functions:**
- `fit_xcorr()`: Fit pulse width from cross-correlation measurements

## Data Formats

### DukeScan Format
TIFF stacks with associated `.log` files containing time delay information.
- Filename format: `*_DS_CH1.tif`, `*_DS_CH2.tif`, etc.
- Log file: `*.log` with `delayArr_ps` field

### Mathematica Format
Binary format with dimensions, time axis, and image data stored as float64 values.

### Pickle Format
Serialized Python dictionary containing:
- `images`: NumPy array of images
- `times`: Array of time delays
- `filename`: Original filename
- `image_dimensions`: Image shape
- `mask`: Boolean mask array

## Example Notebook

See `example.ipynb` for a detailed walkthrough of the package functionality, including:
- Data import from DukeScan format
- Visualization of TA curves and projections
- Phasor analysis workflow
- Masking and thresholding techniques

## Dependencies

- numpy: Numerical operations
- matplotlib: Plotting and visualization
- scikit-image: Image processing and filtering
- scipy: Scientific computing and optimization
- pandas: Data import and manipulation
- scikit-learn: Machine learning (optional, for classification)

See `requirements.txt` for complete list with versions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[Add your license information here]

## Author

@author: david

Created on Thu May 11 14:25:04 2023

## Citation

If you use this software in your research, please cite:
[Add citation information here]

## Support

For questions, issues, or feature requests, please open an issue on the repository.
