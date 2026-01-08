"""
Utility script to convert MATLAB .mat files to NumPy .npz format.

This script converts the parameter data files in the data/ directory
from MATLAB format (.mat) to NumPy format (.npz).

Usage:
    python ConvertData.py
"""

import os
import numpy as np
import scipy.io as sio


def convert_mat_to_npz(mat_file, npz_file):
    """
    Convert a MATLAB .mat file to NumPy .npz format.
    
    Args:
        mat_file: Path to input .mat file
        npz_file: Path to output .npz file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load MATLAB file
        mat_data = sio.loadmat(mat_file)
        
        # Extract params
        if 'params' in mat_data:
            params = mat_data['params'].flatten()
        else:
            print(f"[WARN] No 'params' field found in {mat_file}")
            return False
        
        # Save as NumPy format
        np.savez(npz_file, params=params)
        print(f"[OK] Converted {os.path.basename(mat_file)} -> {os.path.basename(npz_file)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {mat_file}: {e}")
        return False


def main():
    """Convert all .mat files in the data directory to .npz format."""
    data_dir = 'data'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return
    
    # Find all .mat files
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    if not mat_files:
        print(f"[WARN] No .mat files found in {data_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files to convert.")
    print("=" * 70)
    
    success_count = 0
    for mat_file in sorted(mat_files):
        mat_path = os.path.join(data_dir, mat_file)
        npz_file = mat_file.replace('.mat', '.npz')
        npz_path = os.path.join(data_dir, npz_file)
        
        if convert_mat_to_npz(mat_path, npz_path):
            success_count += 1
    
    print("=" * 70)
    print(f"Conversion complete: {success_count}/{len(mat_files)} files converted successfully.")
    
    if success_count == len(mat_files):
        print("All files converted successfully!")
        print("You can now run: python Script.py")
    else:
        print(f"[WARN] {len(mat_files) - success_count} files failed to convert.")


if __name__ == '__main__':
    main()
