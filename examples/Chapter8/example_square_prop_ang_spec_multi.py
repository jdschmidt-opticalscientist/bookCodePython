import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, ang_spec_multi_prop_vac, fresnel_prop_square_ap

# example_square_prop_ang_spec_multi.py
D1 = 2e-3           # diameter of the source aperture [m]
D2 = 6e-3           # diameter of the observation aperture [m]
wvl = 1e-6          # optical wavelength [m]
k = 2 * np.pi / wvl # optical wavenumber [rad/m]
z = 2.0             # propagation distance [m]
delta1 = D1 / 30    # source-plane grid spacing [m]
deltan = D2 / 30    # observation-plane grid spacing [m]
N = 128             # number of grid points
n = 5               # number of partial propagations
# switch from total distance to individual distances
z = np.arange(1, n + 1) * z / n
# source-plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1 / D1) * rect(y1 / D1)
x2, y2, Uout = ang_spec_multi_prop_vac(ap, wvl, delta1, deltan, z)

# Analytic result for y2=0 slice
x2_slice = x2[N // 2, :]
x2_slice_mm = x2_slice * 1e3
y2_val = 0
Dz = z[n-1] # switch back to total distance
Uout_an = fresnel_prop_square_ap(x2_slice, y2_val, D1, wvl, Dz)

# --- Visualization ---
plt.figure(figsize=(12, 5))

# Irradiance Plot
plt.subplot(121)
plt.plot(x2_slice_mm, np.abs(Uout_an)**2, 'rs-', label='Analytic')
plt.plot(x2_slice_mm, np.abs(Uout[N // 2, :])**2, 'bx', label='Numerical')
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
plt.plot(x2_slice_mm, phase_num, 'bx', label='Numerical')
plt.xlim(-5, 5)
plt.title("Square Aperture Diffraction Phase\n($y=0$ slice at $z=1$m)")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("Phase [rad]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()