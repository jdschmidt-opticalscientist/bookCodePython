import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import circ, zernike, ft2, rect, myconv2

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

# Image-plane coordinates
delta_u = wvl * z / (N * delta)
u_vec = np.arange(-N/2, N/2) * delta_u
u, v = np.meshgrid(u_vec, u_vec)

# Object (same coordinates as h)
obj = (rect((u - 1.4e-4) / 5e-5) + 
       rect(u / 5e-5) + 
       rect((u + 1.4e-4) / 5e-5)) * rect(v / 2e-4)

# Convolve the object with the ASF to simulate imaging
img = myconv2(obj, h, 1)

# Visualization
umm = u * 1e3
vmm = v * 1e3
Lmm = L * 1e3

# Set layout here
plt.figure(figsize=(13, 5), layout='constrained') 

plt.subplot(131)
plt.imshow(obj, extent=[umm.min(), umm.max(), vmm.min(), vmm.max()])
plt.xlabel("$x$ [mm]")
plt.ylabel("$y$ [mm]")
plt.title("Object")

plt.subplot(132)
plt.imshow(np.abs(h), extent=[-Lmm/2, Lmm/2, -Lmm/2, Lmm/2])
plt.xlabel("$u$ [mm]")
plt.ylabel("$v$ [mm]")
plt.title("Amplitude Spread Function")

plt.subplot(133)
plt.imshow(np.abs(img)**2, extent=[umm.min(), umm.max(), vmm.min(), vmm.max()])
plt.xlabel("$u$ [mm]")
plt.ylabel("$v$ [mm]")
plt.title("Image Irradiance")

plt.show()