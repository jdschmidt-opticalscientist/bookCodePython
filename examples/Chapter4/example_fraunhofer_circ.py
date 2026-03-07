import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import circ, fraunhofer_prop, jinc

# example_fraunhofer_circ.m
N = 512              # number of grid points per side
L = 7.5e-3           # total size of the grid [m]
d1 = L / N           # source-plane grid spacing [m]
D = 1e-3             # diameter of the aperture [m]
wvl = 1e-6           # optical wavelength [m]
k = 2 * np.pi / wvl
Dz = 20.0            # propagation distance [m]

# Source-plane coordinates and input field
vec1 = np.arange(-N/2, N/2) * d1
x1, y1 = np.meshgrid(vec1, vec1)
Uin = circ(x1, y1, D)

# Perform Fraunhofer propagation
# Note: fraunhofer_prop returns (Uout, x2, y2)
Uout, x2, y2 = fraunhofer_prop(Uin, wvl, d1, Dz)

# Analytic result (Theoretical)
# Intensity pattern follows the Jinc function squared
Uout_th = (np.exp(1j * k / (2 * Dz) * (x2**2 + y2**2)) 
           / (1j * wvl * Dz) * (D**2 * np.pi / 4) 
           * jinc(D * np.sqrt(x2**2 + y2**2) / (wvl * Dz)))

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(np.abs(Uout)**2, extent=[x2.min(), x2.max(), y2.min(), y2.max()])
plt.title("Numerical Irradiance (Airy Disk)")
plt.xlabel("x [m]")

plt.subplot(122)
plt.plot(x2[N//2, :], np.abs(Uout[N//2, :])**2, 'bo', label='Numerical')
plt.plot(x2[N//2, :], np.abs(Uout_th[N//2, :])**2, 'r-', label='Analytic')
plt.title("Horizontal Profile Comparison")
plt.legend()
plt.show()