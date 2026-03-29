# zernikeTest.py

import numpy as np
from scipy.special import gamma

def _zrf(n, m, r):
    R = np.zeros_like(r, dtype=float)
    # n-m must be even and non-negative
    if (n - m) % 2 != 0 or n < m:
        return R
    
    for s in range(int((n - m) / 2) + 1):
        num = (-1)**s * gamma(n - s + 1)
        denom = (gamma(s + 1) * gamma((n + m) / 2 - s + 1) * gamma((n - m) / 2 - s + 1))
        R += num / denom * r**(n - 2 * s)
    return R

def noll_to_nm(j):
    """Corrected Noll to (n, m) mapping."""
    n = 0
    j1 = j
    while j1 > (n + 1):
        j1 -= (n + 1)
        n += 1
    
    # The number of modes up to radial degree n-1 is n(n+1)/2
    # But Noll uses a specific pyramid. Let's use the explicit Noll formula:
    n = int((np.sqrt(8 * j - 7) - 1) / 2)
    m_abs = j - n * (n + 1) // 2 - 1
    
    # Correcting parity and m value
    if n % 2 == 0:
        m = 2 * int(m_abs / 2)
    else:
        m = 2 * int((m_abs + 1) / 2) - 1
    
    if j % 2 > 0:
        m = -m
    return n, m

def zernike(j, r, theta):
    n, m_val = noll_to_nm(j)
    m = abs(m_val)
    # Use a small epsilon for the boundary to avoid floating point clipping
    r_safe = np.where(r <= 1.000001, r, 0.0)
    
    if m == 0:
        return np.sqrt(n + 1) * _zrf(n, 0, r_safe)
    elif j % 2 == 0:
        return np.sqrt(2 * (n + 1)) * _zrf(n, m, r_safe) * np.cos(m * theta)
    else:
        return np.sqrt(2 * (n + 1)) * _zrf(n, m, r_safe) * np.sin(m * theta)

# --- RE-RUN VALIDATION ---
print("--- Revised Zernike Validation ---")
n4, m4 = noll_to_nm(4)
print(f"j=4 maps to n={n4}, m={m4}")
val_4 = zernike(4, np.array([1.0]), np.array([0.0]))[0]
print(f"j=4 (Defocus) at r=1.0: {val_4:.5f} (Target: 1.73205)")

n100, m100 = noll_to_nm(100)
print(f"j=100 maps to n={n100}, m={m100}")
val_100 = zernike(100, np.array([1.0]), np.array([0.0]))[0]
print(f"j=100 at r=1.0: {val_100:.5f} (Target: ~5.099)")