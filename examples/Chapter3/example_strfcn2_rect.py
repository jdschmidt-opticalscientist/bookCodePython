import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, tri, str_fcn2_ft

# example_strfcn2_rect.m
N = 256         # number of samples
L = 16.0        # grid size [m]
delta = L / N   # sample spacing [m]

x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)

w = 2.0         # width of rectangle
# Define 2D signal (Square aperture)
A = rect(x / w) * rect(y / w)
mask = np.ones((N, N))

# Perform discrete 2D structure function calculation
# str_fcn2_ft returns (D, idx)
D_samp, idx = str_fcn2_ft(A, mask, delta)

# Normalize by delta^2 as per the original MATLAB script
C = D_samp / delta**2

# Analytic structure function: 2 * [Var(A) - Corr(A,A)]
# For a binary mask, the variance/autocorrelation leads to the triangle form
C_cont = 2 * (w**2) * (1 - tri(x / w) * tri(y / w))

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(C, extent=[-L/2, L/2, -L/2, L/2])
plt.title("Numerical Structure Function")
plt.colorbar()

plt.subplot(122)
plt.imshow(C_cont, extent=[-L/2, L/2, -L/2, L/2])
plt.title("Analytic Structure Function")
plt.colorbar()
plt.show()