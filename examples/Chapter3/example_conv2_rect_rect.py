import numpy as np
import matplotlib.pyplot as plt
# from wave_prop import rect, tri, myconv2

# example_conv2_rect_rect.m
N = 256         # number of samples
L = 16.0        # grid size [m]
delta = L / N   # sample spacing [m]

x_vec = np.arange(-N/2, N/2) * delta
x, y = np.meshgrid(x_vec, x_vec)

w = 2.0         # width of rectangle
# Define 2D signals (Square apertures)
A = rect(x / w) * rect(y / w)
B = A

# Perform discrete 2D convolution
C = myconv2(A, B, delta)

# Analytic 2D convolution (Product of triangles)
C_cont = (w**2) * tri(x / w) * tri(y / w)

# Optional: Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(np.real(C), extent=[-L/2, L/2, -L/2, L/2])
plt.title('Discrete 2D Convolution')
plt.subplot(122)
plt.imshow(C_cont, extent=[-L/2, L/2, -L/2, L/2])
plt.title('Analytic Solution')
plt.show()