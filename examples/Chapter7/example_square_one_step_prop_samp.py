import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, one_step_prop, fresnel_prop_square_ap

# example_square_one_step_prop_samp.py
D1 = 2e-3           # diam of source aperture [m]
D2 = 3e-3           # diam of observation ROI [m]
delta1 = D1 / 50    # want at least 50 grid pts across ap
wvl = 1e-6          # optical wavelength [m]
Dz = 0.5            # propagation distance [m]
# minimum number of grid points
Nmin = (D1 * wvl * Dz) / (delta1 * (wvl * Dz - D2 * delta1))
N = int(2**np.ceil(np.log2(abs(Nmin))))
# Source plane
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1 / D1) * rect(y1 / D1)
# Simulate Propagation
x2, y2, Uout = one_step_prop(ap, wvl, delta1, Dz)

# Analytic result for y2=0 slice
x2_slice = x2[N // 2, :]
x2_slice_mm = x2_slice * 1e3
y2_val = 0
Uout_an = fresnel_prop_square_ap(x2_slice, y2_val, D1, wvl, Dz)

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