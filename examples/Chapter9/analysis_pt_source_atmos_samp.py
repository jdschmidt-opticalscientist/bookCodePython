import numpy as np
import matplotlib.pyplot as plt

# analysis_pt_source_atmos_samp.py

# 1. Inputs (inherited from previous setup)
# D1, D2, wvl, Dz, r0sw, R, N = 512 (assumed from previous steps)
c = 2
D1p = D1 + c * wvl * Dz / r0sw
D2p = D2 + c * wvl * Dz / r0sw

# 2. Grid for delta analysis
d1_vec = np.linspace(1e-6, 1.1 * wvl * Dz / D2p, 100)
dn_vec = np.linspace(1e-6, 1.1 * wvl * Dz / D1p, 100)
delta1, deltan = np.meshgrid(d1_vec, dn_vec)

# Constraint 1: Geometric limit (Boundary of the angular spectrum)
deltan_max = -D2p / D1p * d1_vec + wvl * Dz / D1p

# Constraint 3: Sampling the quadratic phase (Nyquist limits)
dn_min3 = (1 + Dz / R) * d1_vec - wvl * Dz / D1p
dn_max3 = (1 + Dz / R) * d1_vec + wvl * Dz / D1p

# Constraint 2: Required grid size N
# This calculates the N required for every combination of delta1 and deltan
N_req = (wvl * Dz + D1p * deltan + D2p * delta1) / (2 * delta1 * deltan)

# Constraint 4: Partial Propagation Distance (zmax)
# For a multi-screen simulation, we check how far we can step safely
d1_test, d2_test = 10e-3, 10e-3
N_fixed = 512
zmax = min(d1_test, d2_test)**2 * N_fixed / wvl
nmin = np.ceil(Dz / zmax) + 1

print(f"Max partial propagation distance (zmax): {zmax/1000:.2f} km")
print(f"Minimum number of screens required: {nmin}")

# Visualization of the Valid Sampling Region
plt.figure(figsize=(8, 6))
plt.plot(d1_vec * 1e3, deltan_max * 1e3, 'r', label='Constraint 1 (Overlap)')
plt.plot(d1_vec * 1e3, dn_min3 * 1e3, 'g--', label='Constraint 3 (Min)')
plt.plot(d1_vec * 1e3, dn_max3 * 1e3, 'g', label='Constraint 3 (Max)')
plt.xlabel("delta 1 [mm]")
plt.ylabel("delta n [mm]")
plt.title("Sampling Constraint Map")
plt.legend()
plt.grid(True)
plt.show()