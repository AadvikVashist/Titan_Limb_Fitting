import numpy as np
import matplotlib.pyplot as plt

# # Define the matrix A and the initial vector x0
# angle = 5
# scale = 1.01
# angle = np.deg2rad(angle)
# A = np.array([[np.cos(angle)*scale, np.sin(angle)], [-1*np.sin(angle), np.cos(angle)*scale]])
A = np.array([[1,0], [0,1]])*np.array([[0.8,0.6], [-0.6,0.8]])*np.linalg.inv([[1,0], [0,1]])
x0 = np.array([[3], [0]])

# Initialize a list to hold x0, x1, x2, ...
x_vectors = [x0]

# Calculate x1, x2, ..., by applying A repeatedly
for _ in range(2000):  # Calculate up to x20 for illustration
    x_next = A.dot(x_vectors[-1])
    x_vectors.append(x_next)

# Plotting
x_vectors = np.array(x_vectors).reshape(-1, 2)  # Reshape to 2D array
plt.figure(figsize=(8, 8))
plt.plot( x_vectors[:,0], x_vectors[:,1], label=f'x{1}')  # Adjusted indexing to access scalar values
size = 10
plt.xlim(-1*size, 1*size)
plt.ylim(-1*size, 1*size)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title('Plot of Ax0, Ax1, Ax2, ...')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
