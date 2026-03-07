import numpy as np
# from wave_prop import ang_spec_multi_prop

# example_pt_source_vac_prop.py

# 1. Setup (using variables from previous scripts)
delta1 = d1  # Source spacing [m]
deltan = d2  # Observation spacing [m]
n = nscr     # Number of planes

# 2. Source-plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
r1 = np.sqrt(x1**2 + y1**2)

# 3. Create Point Source
# Band-limited sinc with Gaussian apodization and quadratic phase
pt = (np.exp(-1j * k / (2 * R) * r1**2) / D1**2 * np.sinc(x1/D1) * np.sinc(y1/D1) * np.exp(-(r1 / (4 * D1))**2))

# 4. Define propagation planes (cumulative distances)
z_planes = np.arange(1, n) * Dz / (n - 1)

# 5. Super-Gaussian Absorbing Mask (16th order)
# Prevents aliasing/reflection from grid boundaries
sg = np.exp(-(x1 / (0.47 * N * d1))**16) * np.exp(-(y1 / (0.47 * N * d1))**16)

# Create a stack of masks for the multi-prop function
t = np.repeat(sg[:, :, np.newaxis], n, axis=2)

# 6. Multi-plane propagation
xn, yn, Uvac = ang_spec_multi_prop(pt, wvl, delta1, deltan, z_planes, t)

# 7. Collimate the beam (remove quadratic phase at the observation plane)
Uvac = Uvac * np.exp(-1j * np.pi / (wvl * R) * (xn**2 + yn**2))