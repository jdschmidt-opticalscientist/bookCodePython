import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, ang_spec_prop, fresnel_prop_square_ap

# example_square_prop_ang_spec.m
N = 1024
L = 1e-2
delta1 = L / N
D = 2e-3
wavelength = 1e-6
k = 2 * np.pi / wavelength
Dz = 1.0

# Source plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)

# Square aperture
ap = rect(x1 / D) * rect(y1 / D)

# Scaling the observation plane
delta2 = (wavelength * Dz) / (N * delta1)

# Numerical propagation
x2, y2, Uout = ang_spec_prop(ap, wavelength, delta1, delta2, Dz)

# Analytic result for y2=0 slice
# Note: Python index N//2 corresponds to the center (0 elevation)
x2_slice = x2[N // 2, :]
y2_val = 0
Uout_an = fresnel_prop_square_ap(x2_slice, y2_val, D, wavelength, Dz)

# Plotting the comparison
plt.figure(figsize=(8, 5))
plt.plot(x2_slice, np.abs(Uout[N // 2, :])**2, 'bo', label='Angular Spectrum (Numerical)')
plt.plot(x2_slice, np.abs(Uout_an)**2, 'r-', label='Fresnel Integral (Analytic)')
plt.title("Comparison of Square Aperture Diffraction at $z=1$m")
plt.xlabel("Position [m]")
plt.ylabel("Irradiance")
plt.legend()
plt.show()