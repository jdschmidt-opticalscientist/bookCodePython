import numpy as np
from OpticalWavePropSim import circ, zernike

# example_zernike_projection.m
N = 32         # number of grid points per side
L = 2.0        # total size of the grid [m]
delta = L / N # grid spacing [m]
# xartesian & polar coordinates
vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
# Unit circle aperture
ap = circ(x, y, 2)
# 3 Zernike modes
z2 = zernike(2, r, theta) * ap
z4 = zernike(4, r, theta) * ap
z21 = zernike(21, r, theta) * ap
# create the aberration
W = 0.5 * z2 + 0.25 * z4 - 0.6 * z21
# find only grid points within the aperture
idxAp = ap > 0
# perform linear indexing in column-major order
W_vec = W[idxAp]
Z = np.column_stack((z2[idxAp], z4[idxAp], z21[idxAp]))
# solve the system of equations to compute coefficients
A, residuals, rank, s = np.linalg.lstsq(Z, W_vec, rcond=None)
print(f"Recovered Coefficients: {A}")