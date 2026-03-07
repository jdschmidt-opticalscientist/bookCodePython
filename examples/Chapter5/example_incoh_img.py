import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import circ, zernike, ft2, rect, myconv2

# example_incoh_img.m
N = 256
L = 0.1
D = 0.07
delta = L / N
wvl = 1e-6
z = 0.25

# Pupil-plane coordinates
vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Defocus (Zernike index 4)
zern_map = {4: (2, 0)}
W = 0.05 * zernike(4, 2*r/D, theta, zern_map)

# Complex pupil function and ASF
P = circ(x, y, D) * np.exp(1j * 2 * np.pi * W)
h = ft2(P, delta)

# PSF is the squared magnitude of the ASF
psf = np.abs(h)**2

# Image-plane coordinates
U_grid = wvl * z / (N * delta)
u_vec = np.arange(-N/2, N/2) * U_grid
u, v = np.meshgrid(u_vec, u_vec)

# Object Intensity
obj_intensity = (rect((u - 1.4e-4) / 5e-5) + 
                 rect(u / 5e-5) + 
                 rect((u + 1.4e-4) / 5e-5)) * rect(v / 2e-4)

# Incoherent imaging: convolve intensity with PSF
img_incoherent = myconv2(obj_intensity, psf, 1)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(obj_intensity, extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.title("Object Intensity")
plt.subplot(122)
plt.imshow(np.real(img_incoherent), extent=[u.min(), u.max(), v.min(), v.max()], cmap='gray')
plt.title("Incoherent Image")
plt.show()