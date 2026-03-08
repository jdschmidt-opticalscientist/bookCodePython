import numpy as np
from OpticalWavePropSim import circ, zernike

# example_zernike_projection.m
N = 32         # number of grid points per side
L = 2.0        # total size of the grid [m]
delta = L / N # grid spacing [m]

# Cartesian & polar coordinates
vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Unit circle aperture (radius = 1, so D = 2)
ap = circ(x, y, 2)

# Define Zernike index map (Noll or similar ordering)
# Index 2: Tilt, 4: Defocus, 21: high order
zern_map = {2: (1, 1), 4: (2, 0), 21: (5, 1)}

# 3 Zernike modes
z2 = zernike(2, r, theta, zern_map) * ap
z4 = zernike(4, r, theta, zern_map) * ap
z21 = zernike(21, r, theta, zern_map) * ap

# Create the aberration
W_full = 0.5 * z2 + 0.25 * z4 - 0.6 * z21

# Find only grid points within the aperture
idx = ap > 0
# Perform linear indexing in column-major order
W_vec = W_full[idx]

# Construct the Matrix Z where each column is a Zernike mode
Z = np.column_stack((z2[idx], z4[idx], z21[idx]))

# Solve the linear system: Z * A = W_vec
# np.linalg.lstsq returns (coefficients, residuals, rank, singular_values)
A, residuals, rank, s = np.linalg.lstsq(Z, W_vec, rcond=None)

print(f"Recovered Coefficients: {A}")