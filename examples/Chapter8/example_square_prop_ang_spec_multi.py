import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, ang_spec_multi_prop_vac, fresnel_prop_square_ap

# example_square_prop_ang_spec_multi.m
D1 = 2e-3           # Source diameter [m]
D2 = 6e-3           # Observation diameter [m]
wvl = 1e-6          # Wavelength [m]
k = 2 * np.pi / wvl
z_total = 2.0       # Total distance [m]
delta1 = D1 / 30    # Source spacing
deltan = D2 / 30    # Observation spacing
N = 128             # Grid points
n_steps = 5         # Number of partial propagations

# Create an array of cumulative distances: [0.4, 0.8, 1.2, 1.6, 2.0]
z_steps = np.arange(1, n_steps + 1) * z_total / n_steps

# Source plane
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1 / D1) * rect(y1 / D1)

# Multi-plane propagation
x2, y2, Uout = ang_spec_multi_prop_vac(ap, wvl, delta1, deltan, z_steps)

# Analytic result
mid = N // 2
x2_slice = x2[mid, :]
Uout_an = fresnel_prop_square_ap(x2_slice, 0, D1, wvl, z_total)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(np.abs(Uout)**2, extent=[x2.min(), x2.max(), y2.min(), y2.max()], cmap='viridis')
plt.title(f"Multi-step ASM (n={n_steps})")

plt.subplot(122)
plt.plot(x2_slice, np.abs(Uout[mid, :])**2, 'bo', label='Multi-step Numerical', markersize=4)
plt.plot(x2_slice, np.abs(Uout_an)**2, 'r-', label='Analytic Fresnel')
plt.legend()
plt.show()