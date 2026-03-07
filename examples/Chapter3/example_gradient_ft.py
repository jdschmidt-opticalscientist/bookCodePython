import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import gradient_ft

# example_gradient_ft.m
N = 64          # number of samples
L = 6.0         # grid size [m]
delta = L / N   # grid spacing [m]
x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)
g = np.exp(-(x**2 + y**2))
# Discrete derivatives
gx_samp, gy_samp = gradient_ft(g, delta)
gx_samp = np.real(gx_samp)
gy_samp = np.real(gy_samp)
# Analytic derivatives
gx = -2 * x * np.exp(-(x**2 + y**2))
gy = -2 * y * np.exp(-(x**2 + y**2))
# Subsampling for clarity (plot every 's' points)
s = 2
skip = (slice(None, None, s), slice(None, None, s))

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Numerical Gradient Field
plt.subplot(121)
plt.quiver(x[skip], y[skip], gx_samp[skip], gy_samp[skip], 
           np.sqrt(gx_samp[skip]**2 + gy_samp[skip]**2), cmap='plasma')
plt.title(r'Numerical Gradient $\nabla g$ (FT Method)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='Magnitude')
plt.axis('equal')

# Plot 2: Analytic Gradient Field
plt.subplot(122)
plt.quiver(x[skip], y[skip], gx[skip], gy[skip], 
           np.sqrt(gx[skip]**2 + gy[skip]**2), cmap='plasma')
plt.title(r'Analytic Gradient $\nabla g$')
plt.xlabel('x [m]')
plt.colorbar(label='Magnitude')
plt.axis('equal')

plt.tight_layout()
plt.show()