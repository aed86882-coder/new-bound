#!/bin/bash
# Quick setup script for fast_mm_python

echo "========================================"
echo "Fast MM Python - Quick Setup"
echo "========================================"
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✓ Python 3 found"
echo ""

# Install dependencies
echo "[2/4] Installing dependencies..."
pip3 install numpy scipy
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Try: pip3 install --user numpy scipy"
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "[3/4] Verifying installation..."
python3 -c "import numpy; import scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Please check your Python environment."
    exit 1
fi
echo "✓ Import test passed"
echo ""

# Convert data (if .mat files exist)
echo "[4/4] Checking data files..."
if ls data/*.mat 1> /dev/null 2>&1; then
    echo "Found .mat files. Converting to .npz format..."
    python3 ConvertData.py
    echo "✓ Data conversion complete"
else
    echo "ℹ No .mat files found (or already converted)"
fi
echo ""

echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "You can now run:"
echo "  python3 Script.py"
echo ""

