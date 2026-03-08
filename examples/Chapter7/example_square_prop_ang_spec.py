import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, ang_spec_prop, fresnel_prop_square_ap

# example_square_prop_ang_spec.py
D1 = 2e-3           # diam of the source aperture [m]
D2 = 4e-3           # diam of the obs-plane region of interest [m]
wvl = 1e-6          # optical wavelength [m]
Dz = 0.1            # Propagation distance [m]
delta1 = 9.4848e-6  # Source grid spacing [m]
delta2 = 28.1212e-6 # Observation grid spacing [m]
# minimum number of grid points
Nmin = D1/(2*delta1) + D2/(2*delta2) + (wvl*Dz)/(2*delta1*delta2)
# bump N up to the next power of 2 for efficient FFT
N = int(2**np.ceil(np.log2(Nmin)))

# Source plane
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1/D1) * rect(y1/D1)

# Simulate propagation
x2, y2, Uout = ang_spec_prop(ap, wvl, delta1, delta2, Dz)

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