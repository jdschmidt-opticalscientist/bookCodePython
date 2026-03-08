import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import circ, fraunhofer_prop, jinc

# example_fraunhofer_circ.m
N = 512              # number of grid points per side
L = 7.5e-3           # total size of the grid [m]
d1 = L / N           # source-plane grid spacing [m]
D = 1e-3             # diameter of the aperture [m]
wvl = 1e-6           # optical wavelength [m]
k = 2 * np.pi / wvl
Dz = 20.0            # propagation distance [m]

vec1 = np.arange(-N/2, N/2) * d1
x1, y1 = np.meshgrid(vec1, vec1)
Uin = circ(x1, y1, D)
Uout, x2, y2 = fraunhofer_prop(Uin, wvl, d1, Dz)

# Analytic result
Uout_th = (np.exp(1j * k / (2 * Dz) * (x2**2 + y2**2)) 
           / (1j * wvl * Dz) * (D**2 * np.pi / 4) 
           * jinc(D * np.sqrt(x2**2 + y2**2) / (wvl * Dz)))

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
im = plt.imshow(np.abs(Uout)**2*1e3, extent=[x2.min(), x2.max(), y2.min(), y2.max()])
plt.title("Numerical Irradiance")
plt.xlabel("$x_2$ [m]")
plt.ylabel("$y_2$ [m]")
cb = plt.colorbar(im)
cb.set_label(r'Irradiance [mW/m$^2$]')

plt.subplot(122)
plt.plot(x2[N//2, :], np.abs(Uout_th[N//2, :])**2*1e3, 'rs-', label='Analytic')
plt.plot(x2[N//2, :], np.abs(Uout[N//2, :])**2*1e3, 'bx', label='Numerical')
plt.xlabel("$x_2$ [m]")
plt.ylabel("Irradiance [mW/m$^2$]")
plt.title("$y_2=0$ Slice")
plt.grid(True)
plt.legend()
plt.show()