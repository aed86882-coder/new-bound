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

    VerifyOmega(5, 1.00, 'data/K100_2.37155181.npz')

    VerifyAlpha(5, 'data/alpha_0.32133405.npz')

    VerifyMu(5, 'data/mu_0.52766067.npz')



if __name__ == '__main__':
    main()

