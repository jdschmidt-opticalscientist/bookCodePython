import numpy as np
import matplotlib.pyplot as plt
from OpticalWavePropSim import rect, tri, myconv2

# example_conv2_rect_rect.py
N = 256         # number of samples
L = 16.0        # grid size [m]
delta = L / N   # sample spacing [m]

x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)

w = 2.0         # width of rectangle
A = rect(x / w) * rect(y / w) # signal
B = A # signal
C = myconv2(A, B, delta) # Perform discrete 2D convolution
# continuous convolution
C_cont = (w**2) * tri(x / w) * tri(y / w)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(2, 2, 1)
img1 = plt.imshow(C_cont, extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(img1, label='Amplitude')
plt.xlabel('$x [m]$')
plt.ylabel('$y [m]$')
plt.title('Analytic')

plt.subplot(2, 2, 2)
img2 = plt.imshow(np.real(C), extent=[-L/2, L/2, -L/2, L/2])
plt.colorbar(img2, label='Amplitude')
plt.xlabel('$x [m]$')
plt.ylabel('$y [m]$')
plt.title('Numerical')

# Bottom: y=0 Slice Comparison
plt.subplot(2, 1, 2)
# Extract the slice where y = 0
slice_num = np.real(C[int(N/2)+1, :])
slice_ana = C_cont[int(N/2)+1, :]

plt.plot(x_vec, slice_ana, 'rs-', label='Analytic', lw=2)
plt.plot(x_vec, slice_num, 'bx', label='Numerical')

plt.title(r'Cross-section comparison at $y=0$')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()