import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, one_step_prop, fresnel_prop_square_ap

# example_square_prop_one_step.m
N = 1024
L = 1e-2
delta1 = L / N
D = 2e-3
wvl = 1e-6
k = 2 * np.pi / wvl
Dz = 1.0

# Source plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)

# Square aperture
ap = rect(x1 / D) * rect(y1 / D)

# Numerical propagation using one-step method
x2, y2, Uout = one_step_prop(ap, wvl, delta1, Dz)

# Analytic result for y2=0 slice
mid_idx = N // 2
x2_slice = x2[mid_idx, :]
Uout_an = fresnel_prop_square_ap(x2_slice, 0, D, wvl, Dz)

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(x2_slice, np.abs(Uout[mid_idx, :])**2, 'bo', label='One-Step (Numerical)')
plt.plot(x2_slice, np.abs(Uout_an)**2, 'r-', label='Fresnel Integral (Analytic)')
plt.title("One-Step Fresnel Propagation Comparison")
plt.xlabel("Position [m]")
plt.ylabel("Irradiance")
plt.legend()
plt.grid(True)
plt.show()