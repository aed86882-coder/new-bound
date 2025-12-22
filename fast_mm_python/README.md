# New Bounds for Matrix Multiplication: Python Implementation

This is a Python implementation of the optimization and verification code for the paper _New Bounds for Matrix Multiplication: from Alpha to Omega_. The implementation maintains naming consistency with the original MATLAB code while using Python/NumPy/SciPy.

## Overview

This project computes upper bounds on the matrix multiplication exponent ω(κ) and related parameters α and μ. It provides:

- **Verification**: Check provided parameter bounds (ω ≤ 2.371552, α ≥ 0.321334, μ ≤ 0.527661)
- **Automatic Differentiation**: Custom GVar class for gradient computation
- **Parameter Management**: Complete system for handling optimization parameters
- **Modular Design**: Clean separation matching the MATLAB structure

## Requirements

- Python 3.8 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0

## Installation

### Step 1: Clone or Download the Repository

```bash
cd /path/to/fast_mm_code_python
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fast_mm python=3.8
conda activate fast_mm
```

### Step 3: Install Dependencies

```bash
cd fast_mm_python
pip3 install numpy scipy
```

Or use the requirements file:
```bash
pip3 install -r requirements.txt
```

**Verify installation**:
```bash
python3 -c "import numpy; import scipy; print('✓ Dependencies installed!')"
```

### Step 4: Convert Data Files

The MATLAB `.mat` data files in the `data/` directory need to be converted to NumPy `.npz` format:

```bash
python ConvertData.py
```

This will:
- Read all `.mat` files from `data/`
- Convert them to `.npz` format
- Save them in the same `data/` directory

**Note**: The data files are already included in the `data/` directory.

## Quick Start

### Install Dependencies First

```bash
pip3 install numpy scipy
```

### Verify All Bounds

Run the main verification script:

```bash
python3 Script.py
```


