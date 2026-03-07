import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import ang_spec_prop

# example_pt_source_gaussian.m
D = 8e-3            # diameter of observation aperture [m]
wvl = 1e-6          # wavelength [m]
k = 2 * np.pi / wvl
Dz = 1.0            # propagation distance [m]

arg = D / (wvl * Dz)
delta1 = 1 / (10 * arg)   # source-plane grid spacing [m]
delta2 = D / 100          # observation-plane grid spacing [m]
N = 1024                  # grid points

# Source-plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
r1 = np.sqrt(x1**2 + y1**2)

# Band-limited point source with Gaussian apodization
# arg/4 determines the "width" of the Gaussian smoothing
pt = (np.exp(-1j * k / (2 * Dz) * r1**2) * (arg**2) * np.sinc(arg * x1) * np.sinc(arg * y1) * np.exp(-( (arg / 4) * r1 )**2))

# Propagate
x2, y2, Uout = ang_spec_prop(pt, wvl, delta1, delta2, Dz)

# Quick look at the result
# plt.imshow(np.abs(Uout)**2, extent=[x2.min(), x2.max(), y2.min(), y2.max()])
# plt.title("Irradiance at Observation Plane (Gaussian Apodized)")
# plt.show()