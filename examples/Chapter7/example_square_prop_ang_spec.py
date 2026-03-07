import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, ang_spec_prop, fresnel_prop_square_ap

# example_square_prop_ang_spec.m
D1 = 2e-3           # Source diameter [m]
D2 = 4e-3           # Observation diameter [m]
wvl = 1e-6          # Wavelength [m]
Dz = 0.1            # Propagation distance [m]

delta1 = 9.4848e-6  # Source grid spacing [m]
delta2 = 28.1212e-6 # Observation grid spacing [m]

# Calculate minimum N to prevent aliasing and satisfy geometry
Nmin = D1/(2*delta1) + D2/(2*delta2) + (wvl*Dz)/(2*delta1*delta2)
N = int(2**np.ceil(np.log2(Nmin)))

# Source plane coordinates
vec1 = np.arange(-N/2, N/2) * delta1
x1, y1 = np.meshgrid(vec1, vec1)
ap = rect(x1/D1) * rect(y1/D1)

# Perform propagation
x2, y2, Uout = ang_spec_prop(ap, wvl, delta1, delta2, Dz)

# Analytic result for horizontal slice
mid = N // 2
x2_slice = x2[mid, :]
Uout_an = fresnel_prop_square_ap(x2_slice, 0, D1, wvl, Dz)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(np.abs(Uout)**2, extent=[x2.min(), x2.max(), y2.min(), y2.max()], cmap='inferno')
plt.title(f"Numerical Irradiance (N={N})")

plt.subplot(122)
plt.plot(x2_slice, np.abs(Uout[mid, :])**2, 'bo', label='Numerical', markersize=3)
plt.plot(x2_slice, np.abs(Uout_an)**2, 'r-', label='Analytic')
plt.legend()
plt.show()