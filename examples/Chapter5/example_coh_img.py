import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import circ, zernike, ft2, rect, myconv2

# example_coh_img.m
N = 256             # grid points per side
L = 0.1             # total size of the grid [m]
D = 0.07            # diameter of pupil [m]
delta = L / N       # grid spacing [m]
wvl = 1e-6          # wavelength [m]
z = 0.25            # image distance [m]

# Pupil-plane coordinates
vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Define Zernike index map (Index 4: n=2, m=0 for defocus)
zern_map = {4: (2, 0)}

# Wavefront aberration (0.05 waves of defocus)
W = 0.05 * zernike(4, 2*r/D, theta, zern_map)

# Complex pupil function
P = circ(x, y, D) * np.exp(1j * 2 * np.pi * W)

# Amplitude Spread Function (ASF)
h = ft2(P, delta)

# Image-plane coordinates (scaled by wavelength and distance)
delta_u = wvl * z / (N * delta)
u_vec = np.arange(-N/2, N/2) * delta_u
u, v = np.meshgrid(u_vec, u_vec)

# Object: A three-bar target
obj = (rect((u - 1.4e-4) / 5e-5) + 
       rect(u / 5e-5) + 
       rect((u + 1.4e-4) / 5e-5)) * rect(v / 2e-4)

# Convolve the object with the ASF to simulate imaging
# delta is set to 1 here as we are convolving on the current sampling grid
img = myconv2(obj, h, 1)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(131); plt.imshow(np.abs(P), extent=[-L/2, L/2, -L/2, L/2]); plt.title("Pupil Amplitude")
plt.subplot(132); plt.imshow(obj, extent=[u.min(), u.max(), v.min(), v.max()]); plt.title("Object")
plt.subplot(133); plt.imshow(np.abs(img)**2, extent=[u.min(), u.max(), v.min(), v.max()]); plt.title("Coherent Image")
plt.show()