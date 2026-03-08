import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, ang_spec_prop, fresnel_prop_square_ap

# example_square_prop_ang_spec.py
N = 1024       # number of grid points per side
L = 1e-2       # total size of the grid [m]
delta1 = L / N # grid spacing [m]
D = 2e-3       # diameter of the aperture [m]
wvl = 1e-6     # optical wavelength [m]
k = 2 * np.pi / wvl
Dz = 1.0       # propagation distance [m]

vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)

ap = rect(x1 / D) * rect(y1 / D)
delta2 = (wvl * Dz) / (N * delta1)
x2, y2, Uout = ang_spec_prop(ap, wvl, delta1, delta2, Dz)

# Analytic result for y2=0 slice
x2_slice = x2[N // 2, :]
x2_slice_mm = x2_slice * 1e3
y2_val = 0
Uout_an = fresnel_prop_square_ap(x2_slice, y2_val, D, wvl, Dz)

# --- Visualization ---
plt.figure(figsize=(12, 5))

# Irradiance Plot
plt.subplot(121)
plt.plot(x2_slice_mm, np.abs(Uout_an)**2, 'rs-', label='Analytic')
plt.plot(x2_slice_mm, np.abs(Uout[N // 2, :])**2, 'bx-', label='Numerical')
plt.xlim(-5, 5)
plt.title("Square Aperture Diffraction Irradiance\n($y=0$ slice at $z=1$m)")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("Irradiance [W/m$^2$]")
plt.legend()
plt.grid(True)

# Phase Plot
plt.subplot(122)
# Extracting phase from analytic and numerical results
phase_an = np.angle(Uout_an)
phase_num = np.angle(Uout[N // 2, :])

plt.plot(x2_slice_mm, phase_an, 'rs-', label='Analytic')
plt.plot(x2_slice_mm, phase_num, 'bx-', label='Numerical')
plt.xlim(-5, 5)
plt.title("Square Aperture Diffraction Phase\n($y=0$ slice at $z=1$m)")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("Phase [rad]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()