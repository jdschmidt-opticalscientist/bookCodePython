import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import gradient_ft

# example_gradient_ft.m
N = 64          # number of samples
L = 6.0         # grid size [m]
delta = L / N   # grid spacing [m]
x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)

# 2D Gaussian function
g = np.exp(-(x**2 + y**2))

# Discrete derivatives using FT property
gx_samp, gy_samp = gradient_ft(g, delta)
gx_samp = np.real(gx_samp)
gy_samp = np.real(gy_samp)

# Analytic derivatives
gx = -2 * x * np.exp(-(x**2 + y**2))
gy = -2 * y * np.exp(-(x**2 + y**2))

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(gx_samp, extent=[-L/2, L/2, -L/2, L/2])
plt.title(r'Numerical $\partial g / \partial x$')
plt.colorbar()

plt.subplot(122)
plt.imshow(gx, extent=[-L/2, L/2, -L/2, L/2])
plt.title(r'Analytic $\partial g / \partial x$')
plt.colorbar()
plt.show()