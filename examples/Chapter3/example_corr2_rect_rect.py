import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, tri, corr2_ft

# example_corr2_rect_rect.m
N = 256         # number of samples
L = 16.0        # grid size [m]
delta = L / N   # sample spacing [m]

x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)

w = 2.0         # width of rectangle
# Define 2D signal (Square aperture)
A = rect(x / w) * rect(y / w)
mask = np.ones((N, N))

# Perform discrete 2D correlation
# Note: corr2_ft returns (correlation_result, mask_index)
C, idx = corr2_ft(A, A, mask, delta)

# Analytic correlation (Product of triangles for symmetric rects)
C_cont = (w**2) * tri(x / w) * tri(y / w)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(np.real(C), extent=[-L/2, L/2, -L/2, L/2])
plt.title('Discrete 2D Correlation')
plt.colorbar()

plt.subplot(122)
plt.imshow(C_cont, extent=[-L/2, L/2, -L/2, L/2])
plt.title('Analytic Solution')
plt.colorbar()
plt.show()