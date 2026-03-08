# -*- coding: utf-8 -*-
# Copyright (c) 2026, [Jason D. Schmidt]
# All rights reserved.
#
# Licensed under the BSD 3-Clause License. 
# See LICENSE file in the project root for full license information.

import math
import numpy as np
from scipy.special import j1
from scipy.special import fresnel # Returns (S, C)
from scipy.special import gamma


def ft(g, delta):
    """1D Fourier Transform"""
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(g))) * delta

def ift(G, delta_f):
    """1D Inverse Fourier Transform"""
    return np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G))) * (len(G) * delta_f)

def ft2(g, delta):
    """2D Fourier Transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * (delta**2)

def ift2(G, delta_f):
    """2D Inverse Fourier Transform"""
    N = G.shape[0]
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N * delta_f)**2

def rect(x, D=1.0):
    """Rectangle function"""
    x = np.abs(x)
    y = (x < D/2).astype(float)
    y[x == D/2] = 0.5
    return y

def tri(t):
    """Triangle function"""
    t = np.abs(t)
    y = np.zeros_like(t)
    idx = t < 1.0
    y[idx] = 1.0 - t[idx]
    return y

def circ(x, y, D=1.0):
    """function z = circ(x, y, D)"""
    r = np.sqrt(x**2 + y**2)
    z = (r < D/2).astype(float)
    # Handle the boundary r = D/2
    z[r == D/2] = 0.5
    return z

def jinc(x):
    """Jinc function: 2*J1(pi*x) / (pi*x)"""
    y = np.ones_like(x, dtype=float)
    mask = x != 0
    y[mask] = 2.0 * j1(np.pi * x[mask]) / (np.pi * x[mask])
    return y
    
def myconv2(A, B, delta):
    """2D Convolution using FFT"""
    N = A.shape[0]
    return ift2(ft2(A, delta) * ft2(B, delta), 1 / (N * delta))

def ang_spec_multi_prop(Uin, wvl, delta1, deltan, z, t):
    """Angular spectrum multi-plane propagation with transmission screens"""
    N = Uin.shape[0]
    vec = np.arange(-N/2, N/2)
    nx, ny = np.meshgrid(vec, vec)
    k = 2 * np.pi / wvl
    
    # Super-Gaussian absorbing boundary
    nsq = nx**2 + ny**2
    w = 0.47 * N
    sg = np.exp(-nsq**8 / w**16)
    
    # Setup propagation planes
    z_arr = np.concatenate(([0], np.atleast_1d(z)))
    n = len(z_arr)
    Delta_z = np.diff(z_arr)
    
    alpha = z_arr / z_arr[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:-1]
    
    # Initial phase and transmission
    x1, y1 = nx * delta[0], ny * delta[0]
    r1sq = x1**2 + y1**2
    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * r1sq)
    Uin = Uin * Q1 * t[:, :, 0]
    
    dz = 0.0 # Define in scope for final Q3
    for idx in range(n - 1):
        deltaf = 1 / (N * delta[idx])
        fsq = (nx * deltaf)**2 + (ny * deltaf)**2
        dz = Delta_z[idx]
        
        # Quadratic phase factor
        Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m[idx] / k * fsq)
        # Propagate and apply next screen
        Uin = sg * t[:, :, idx+1] * ift2(Q2 * ft2(Uin / m[idx], delta[idx]), deltaf)
        
    # Observation plane coordinates
    xn, yn = nx * delta[-1], ny * delta[-1]
    rnsq = xn**2 + yn**2
    Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * dz) * rnsq)
    Uout = Q3 * Uin
    return xn, yn, Uout

def ang_spec_multi_prop_vac(Uin, wvl, delta1, deltan, z):
    """Angular spectrum multi-plane propagation in vacuum (no screens)"""
    # Identical to above but skipping 't' multiplications
    N = Uin.shape[0]
    vec = np.arange(-N/2, N/2)
    nx, ny = np.meshgrid(vec, vec)
    k = 2 * np.pi / wvl
    nsq = nx**2 + ny**2
    sg = np.exp(-nsq**8 / (0.47*N)**16)
    
    z_arr = np.concatenate(([0], np.atleast_1d(z)))
    Delta_z = np.diff(z_arr)
    alpha = z_arr / z_arr[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:-1]
    
    Uin = Uin * np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * (nx*delta[0])**2 + (ny*delta[0])**2)
    
    dz = 0.0
    for idx in range(len(z_arr) - 1):
        df = 1 / (N * delta[idx])
        fsq = (nx * df)**2 + (ny * df)**2
        dz = Delta_z[idx]
        Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m[idx] / k * fsq)
        Uin = sg * ift2(Q2 * ft2(Uin / m[idx], delta[idx]), df)
        
    xn, yn = nx * delta[-1], ny * delta[-1]
    Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * dz) * (xn**2 + yn**2))
    return xn, yn, Q3 * Uin

def ang_spec_prop(Uin, wvl, d1, d2, Dz):
    """Single step angular spectrum propagation with scaling"""
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    vec = np.arange(-N/2, N/2)
    nx, ny = np.meshgrid(vec, vec)
    
    df1 = 1 / (N * d1)
    m = d2 / d1
    
    Q1 = np.exp(1j * k / 2 * (1 - m) / Dz * ((nx*d1)**2 + (ny*d1)**2))
    Q2 = np.exp(-1j * np.pi**2 * 2 * Dz / m / k * ((nx*df1)**2 + (ny*df1)**2))
    Q3 = np.exp(1j * k / 2 * (m - 1) / (m * Dz) * ((nx*d2)**2 + (ny*d2)**2))
    
    Uout = Q3 * ift2(Q2 * ft2(Q1 * Uin / m, d1), df1)
    return nx * d2, ny * d2, Uout
    
def ang_spec_propABCD(Uin, wvl, d1, d2, ABCD):
    # function [x2 y2 Uout] ...
    #     = ang_spec_propABCD(Uin, wwl, d1, d2, ABCD)
    N = Uin.shape[0]   # assume square grid
    # source-plane coordinates
    vec1 = np.arange(-N/2, N/2) * d1
    x1, y1 = np.meshgrid(vec1, vec1)
    r1sq = x1**2 + y1**2
    # spatial frequencies (of source plane)
    df1 = 1 / (N*d1)
    vecf = np.arange(-N/2, N/2) * df1
    fX, fY = np.meshgrid(vecf, vecf)
    fsq = fX**2 + fY**2
    # scaling parameter
    m = d2/d1
    # observation-plane coordinates
    vec2 = np.arange(-N/2, N/2) * d2
    x2, y2 = np.meshgrid(vec2, vec2)
    r2sq = x2**2 + y2**2
    # optical system matrix
    A = ABCD[0,0]; B = ABCD[0,1]; D = ABCD[1,1]
    # quadratic phase factors
    Q1 = np.exp(1j*np.pi/(wvl*B)*(A-m)*r1sq)
    Q2 = np.exp(-1j*np.pi*wvl*B/m*fsq)
    Q3 = np.exp(1j*np.pi/(wvl*B)*(D-1/m)*r2sq)
    # compute the propagated field
    Uout = Q3 * ift2(Q2 * ft2(Q1 * Uin / m, d1), df1)
    
    return x2, y2, Uout

def corr2_ft(u1, u2, mask, delta):
    """2D Correlation using FFT with mask normalization"""
    N = u1.shape[0]
    delta_f = 1 / (N * delta)
    
    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    U12corr = ift2(np.conj(U1) * U2, delta_f)
    
    areamask = np.sum(mask) * delta**2
    # Mask autocorrelation for normalization
    maskcorr = ift2(np.abs(ft2(mask, delta))**2, delta_f) / areamask
    
    # Avoid division by zero/very small values
    idx = maskcorr >= (delta**2 / areamask)
    c = np.zeros((N, N), dtype=complex)
    c[idx] = U12corr[idx] / maskcorr[idx]
    
    return c, idx

def derivative_ft(g, delta, n):
    """n-th order derivative using Fourier Transform property"""
    N = len(g)
    F = 1 / (N * delta)
    f_X = np.arange(-N/2, N/2) * F
    
    # (i*2*pi*f)^n * FT(g)
    return ift((1j * 2 * np.pi * f_X)**n * ft(g, delta), F)

def fraunhofer_prop(Uin, wvl, d1, Dz):
    """Fraunhofer (far-field) propagation"""
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    fX = np.arange(-N/2, N/2) / (N * d1)
    
    # Observation-plane coordinates
    v = wvl * Dz * fX
    x2, y2 = np.meshgrid(v, v)
    
    Uout = (np.exp(1j * k / (2 * Dz) * (x2**2 + y2**2)) 
            / (1j * wvl * Dz) * ft2(Uin, d1))
    return Uout, x2, y2

def fresnel_prop_square_ap(x2, y2, D1, wvl, Dz):
    """Analytic Fresnel propagation for a square aperture"""
    N_F = (D1 / 2)**2 / (wvl * Dz)
    
    bigX = x2 / np.sqrt(wvl * Dz)
    bigY = y2 / np.sqrt(wvl * Dz)
    
    # Normalized coordinates
    alpha1 = -np.sqrt(2) * (np.sqrt(N_F) + bigX)
    alpha2 =  np.sqrt(2) * (np.sqrt(N_F) - bigX)
    beta1  = -np.sqrt(2) * (np.sqrt(N_F) + bigY)
    beta2  =  np.sqrt(2) * (np.sqrt(N_F) - bigY)
    
    # scipy.special.fresnel returns (S, C)
    sa1, ca1 = fresnel(alpha1)
    sa2, ca2 = fresnel(alpha2)
    sb1, cb1 = fresnel(beta1)
    sb2, cb2 = fresnel(beta2)
    
    U = 1 / (2j) * ((ca2 - ca1) + 1j * (sa2 - sa1)) * \
                   ((cb2 - cb1) + 1j * (sb2 - sb1))
    return U
    
def ft_phase_screen(r0, N, delta, L0, l0):
    """
    Generate a phase screen using the FFT method with a 
    modified von Karman PSD.
    """
    del_f = 1 / (N * delta)
    fx = np.arange(-N/2, N/2) * del_f
    FX, FY = np.meshgrid(fx, fx)
    
    # Polar coordinates
    f = np.sqrt(FX**2 + FY**2)
    fm = 5.92 / (l0 * 2 * np.pi)
    f0 = 1 / L0
    
    # Phase PSD
    PSD_phi = 0.023 * r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
    PSD_phi[N//2, N//2] = 0 # Zero out DC component
    
    # Random Fourier coefficients
    cn = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * np.sqrt(PSD_phi) * del_f
    return np.real(ift2(cn, 1))

def ft_sh_phase_screen(r0, N, delta, L0, l0):
    """
    Generate phase screen with subharmonics to compensate for 
    low-frequency undersampling in the FFT method.
    """
    D = N * delta
    phz_hi = ft_phase_screen(r0, N, delta, L0, l0)
    
    vec = np.arange(-N/2, N/2) * delta
    x, y = np.meshgrid(vec, vec)
    phz_lo = np.zeros((N, N), dtype=complex)
    
    # Loop over 3 subharmonic levels
    for p in range(1, 4):
        del_f = 1 / (3**p * D)
        fx_vec = np.arange(-1, 2) * del_f
        fx, fy = np.meshgrid(fx_vec, fx_vec)
        f = np.sqrt(fx**2 + fy**2)
        
        fm = 5.92 / (l0 * 2 * np.pi)
        f0 = 1 / L0
        PSD_phi = 0.023 * r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
        PSD_phi[1, 1] = 0 # Center (DC)
        
        cn = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)) * np.sqrt(PSD_phi) * del_f
        
        SH = np.zeros((N, N), dtype=complex)
        for ii in range(9): # 3x3 grid
            SH += cn.flat[ii] * np.exp(1j * 2 * np.pi * (fx.flat[ii] * x + fy.flat[ii] * y))
        phz_lo += SH
        
    phz_lo = np.real(phz_lo)
    phz_lo -= np.mean(phz_lo)
    return phz_lo, phz_hi

def gradient_ft(g, delta):
    """Compute the gradient of a 2D array using the FT property."""
    N = g.shape[0]
    F = 1 / (N * delta)
    f_vec = np.arange(-N/2, N/2) * F
    fX, fY = np.meshgrid(f_vec, f_vec)
    
    G = ft2(g, delta)
    gx = ift2(1j * 2 * np.pi * fX * G, F)
    gy = ift2(1j * 2 * np.pi * fY * G, F)
    return gx, gy

def lens_against_ft(Uin, wvl, d1, f):
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    fX_vec = np.arange(-N/2, N/2) / (N * d1)
    x2, y2 = np.meshgrid(wvl * f * fX_vec, wvl * f * fX_vec)
    
    Uout = (np.exp(1j * k / (2 * f) * (x2**2 + y2**2)) 
            / (1j * wvl * f) * ft2(Uin, d1))
    return Uout, x2, y2

def lens_behind_ft(Uin, wvl, d1, f, d):
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    fX_vec = np.arange(-N/2, N/2) / (N * d1)
    x2, y2 = np.meshgrid(wvl * d * fX_vec, wvl * d * fX_vec)
    
    Uout = (f/d * 1 / (1j * wvl * d) 
            * np.exp(1j * k / (2 * d) * (x2**2 + y2**2)) * ft2(Uin, d1))
    return x2, y2, Uout

def lens_in_front_ft(Uin, wvl, d1, f, d):
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    fX_vec = np.arange(-N/2, N/2) / (N * d1)
    x2, y2 = np.meshgrid(wvl * f * fX_vec, wvl * f * fX_vec)
    
    Uout = (1 / (1j * wvl * f) 
            * np.exp(1j * k / (2 * f) * (1 - d/f) * (x2**2 + y2**2)) 
            * ft2(Uin, d1))
    return x2, y2, Uout

def myconv(A, B, delta):
    """1D convolution using FFT"""
    N = len(A)
    return ift(ft(A, delta) * ft(B, delta), 1 / (N * delta))
    
def one_step_prop(Uin, wvl, d1, Dz):
    """One-step Fresnel propagation (Fresnel-Kirchhoff integral)."""
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    
    vec1 = np.arange(-N/2, N/2) * d1
    x1, y1 = np.meshgrid(vec1, vec1)
    
    vec2 = np.arange(-N/2, N/2) / (N * d1) * wvl * Dz
    x2, y2 = np.meshgrid(vec2, vec2)
    
    Uout = (1 / (1j * wvl * Dz) * np.exp(1j * k / (2 * Dz) * (x2**2 + y2**2)) * ft2(Uin * np.exp(1j * k / (2 * Dz) * (x1**2 + y1**2)), d1))
    return x2, y2, Uout

def str_fcn2_ft(ph, mask, delta):
    """2D Phase structure function using FFT."""
    N = ph.shape[0]
    ph = ph * mask
    
    P = ft2(ph, delta)
    S = ft2(ph**2, delta)
    W = ft2(mask, delta)
    
    delta_f = 1 / (N * delta)
    w2 = ift2(W * np.conj(W), delta_f)
    
    # Structure function calculation
    D = 2 * ift2(np.real(S * np.conj(W)) - np.abs(P)**2, delta_f) / w2
    
    areamask = np.sum(mask) * delta**2
    maskcorr = ift2(np.abs(ft2(mask, delta))**2, delta_f) / areamask
    idx = maskcorr >= (delta**2 / areamask)
    
    D_real = np.real(D)
    D_real[~idx] = 0
    return D_real, idx

def two_step_prop(Uin, wvl, d1, d2, Dz):
    """Two-step Fresnel propagation for arbitrary magnification."""
    N = Uin.shape[0]
    k = 2 * np.pi / wvl
    m = d2 / d1
    
    # Source coordinates
    vec1 = np.arange(-N/2, N/2) * d1
    x1, y1 = np.meshgrid(vec1, vec1)
    
    # Intermediate plane
    Dz1 = Dz / (1 - m)
    d1a = wvl * np.abs(Dz1) / (N * d1)
    vec1a = np.arange(-N/2, N/2) * d1a
    x1a, y1a = np.meshgrid(vec1a, vec1a)
    
    Uitm = (1 / (1j * wvl * Dz1) * np.exp(1j * k / (2 * Dz1) * (x1a**2 + y1a**2)) * ft2(Uin * np.exp(1j * k / (2 * Dz1) * (x1**2 + y1**2)), d1))
    
    # Observation plane
    Dz2 = Dz - Dz1
    vec2 = np.arange(-N/2, N/2) * d2
    x2, y2 = np.meshgrid(vec2, vec2)
    
    Uout = (1 / (1j * wvl * Dz2) * np.exp(1j * k / (2 * Dz2) * (x2**2 + y2**2)) * ft2(Uitm * np.exp(1j * k / (2 * Dz2) * (x1a**2 + y1a**2)), d1a))
    return x2, y2, Uout

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
