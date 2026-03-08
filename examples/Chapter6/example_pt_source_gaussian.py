import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import ang_spec_prop

# example_pt_source_gaussian.py
D = 8e-3            # diameter of observation aperture [m]
wvl = 1e-6          # wavelength [m]
k = 2 * np.pi / wvl # optical wavenumber [rad/m]
Dz = 1.0            # propagation distance [m]

arg = D / (wvl * Dz)
delta1 = 1 / (10 * arg)   # source-plane grid spacing [m]
delta2 = D / 100          # observation-plane grid spacing [m]
N = 1024                  # number of grid points
# source-plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
r1 = np.sqrt(x1**2 + y1**2)
pt = (np.exp(-1j * k / (2 * Dz) * r1**2) * (arg**2) * np.sinc(arg * x1) * np.sinc(arg * y1) * np.exp(-( (arg / 4) * r1 )**2))
x2, y2, Uout = ang_spec_prop(pt, wvl, delta1, delta2, Dz)

# --- Visualization ---

# Scale coordinates to mm
mid = N // 2
x_slice = x2[mid, :]
x2_mm = x2 * 1e3
y2_mm = y2 * 1e3
x_slice_mm = x_slice * 1e3

plt.figure(figsize=(15, 5))

# Propagated Irradiance (2D)
plt.subplot(131)
I_out = np.abs(Uout)**2
plt.imshow(I_out, extent=[x2_mm.min(), x2_mm.max(), y2_mm.min(), y2_mm.max()], cmap='gray')
plt.title("Numerically Propagated\nPoint-Source Irradiance")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("$y_2$ [mm]")
plt.colorbar(label='Irradiance')

# y=0 Slice of Irradiance
I_slice = I_out[mid, :]

plt.subplot(132)
plt.plot(x_slice_mm, I_slice)
plt.title("Numerically Propagated\nPoint-Source Irradiance Slice")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("Irradiance")
plt.grid(True)

# y=0 Slice of Unwrapped Phase
wrapped_phase = np.angle(Uout[mid, :])
unwrapped_phase = np.unwrap(wrapped_phase)
unwrapped_shifted = unwrapped_phase - unwrapped_phase[mid]

expected_phase = (k / (2 * Dz) * x_slice**2)

plt.subplot(133)
plt.plot(x_slice_mm, expected_phase, 'r-', label='Analytic', alpha=0.8)
plt.plot(x_slice_mm, unwrapped_shifted, 'b-.', label='Numerical')
plt.title("Numerically Propagated\nPoint-Source Phase")
plt.xlabel("$x_2$ [mm]")
plt.ylabel("Phase [rad]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()