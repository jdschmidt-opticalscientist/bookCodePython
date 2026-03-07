import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, one_step_prop, fresnel_prop_square_ap

# example_square_one_step_prop_samp.m
D1 = 2e-3           # diameter of source aperture [m]
D2 = 3e-3           # diameter of observation ROI [m]
delta1 = D1 / 50    # source grid spacing
wvl = 1e-6
Dz = 0.5            # propagation distance [m]

# Calculate minimum N to avoid aliasing and cover D2
Nmin = (D1 * wvl * Dz) / (delta1 * (wvl * Dz - D2 * delta1))
N = int(2**np.ceil(np.log2(abs(Nmin))))

# Source plane
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1 / D1) * rect(y1 / D1)

# Propagation
x2, y2, Uout = one_step_prop(ap, wvl, delta1, Dz)

# Analytic comparison
mid = N // 2
x2_slice = x2[mid, :]
Uout_an = fresnel_prop_square_ap(x2_slice, 0, D1, wvl, Dz)

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(np.abs(Uout)**2, extent=[x2.min(), x2.max(), y2.min(), y2.max()])
plt.title(f"Irradiance (N={N})")

plt.subplot(122)
plt.plot(x2_slice, np.abs(Uout[mid, :])**2, 'bo', label='Numerical')
plt.plot(x2_slice, np.abs(Uout_an)**2, 'r-', label='Analytic')
plt.legend()
plt.show()