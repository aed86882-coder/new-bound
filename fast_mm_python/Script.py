"""
Main verification script for matrix multiplication bounds.

This script verifies the provided sets of parameters for omega, alpha, and mu bounds.
"""

from src.verify import VerifyOmega, VerifyAlpha, VerifyMu


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Matrix Multiplication Bounds Verification - Python Implementation")
    print("=" * 70)
    print()
    
    # Verify main bounds
    print("Verifying main bounds:")
    print("-" * 70)
    
    try:
        VerifyOmega(5, 1.00, 'data/K100_2.37155181.npz')
    except FileNotFoundError:
        print("[WARN] Data file not found. Please convert MATLAB .mat files to .npz format.")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    
    print()
    
    try:
        VerifyAlpha(5, 'data/alpha_0.32133405.npz')
    except FileNotFoundError:
        print("[WARN] Data file not found. Please convert MATLAB .mat files to .npz format.")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    
    print()
    
    try:
        VerifyMu(5, 'data/mu_0.52766067.npz')
    except FileNotFoundError:
        print("[WARN] Data file not found. Please convert MATLAB .mat files to .npz format.")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    
    print()
    print("=" * 70)
    print("Verification complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()

