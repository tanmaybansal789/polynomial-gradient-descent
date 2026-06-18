import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

true_coeffs = np.array([1, -15, 85, -225, 274, -120])

with open("roots_history.txt", "r") as f:
    lines = f.readlines()
    history = []
    for line in lines:
        roots = [float(r) for r in line.strip().split(",")]
        history.append(roots)
history = np.array(history)
stuck_roots = history[-1]

def get_coeffs(roots):
    """Helper to compute coefficients from a numpy array of roots."""
    n = len(roots)
    coeffs = []
    for k in range(n + 1):
        combos = combinations(roots, k)
        total = sum(np.prod(c) for c in combos)
        sign = 1 if k % 2 == 0 else -1
        coeffs.append(sign * total)
    return np.array(coeffs)

def compute_loss(roots):
    pred = get_coeffs(roots)
    return np.mean((true_coeffs - pred) ** 2)

# 3. Create two random orthogonal directions in 5D root space
np.random.seed(20)
dir_x = np.random.normal(size=5)
dir_y = np.random.normal(size=5)
# Orthogonalize them
dir_x /= np.linalg.norm(dir_x)
dir_y -= np.dot(dir_y, dir_x) * dir_x
dir_y /= np.linalg.norm(dir_y)

# 4. Generate the grid around our stuck roots
x_range = np.linspace(-5, 5, 50)
y_range = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

for i in range(len(x_range)):
    for j in range(len(y_range)):
        # Shift away from the center point using our 2D grid coordinates
        current_roots = stuck_roots + X[i,j]*dir_x + Y[i,j]*dir_y
        Z[i,j] = compute_loss(current_roots)

# Log-scale the loss so the graph doesn't explode vertically
Z_log = np.log10(Z + 1e-6)

# 5. Plot the beautiful landscape
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_log, cmap='terrain', edgecolor='none', alpha=0.6)

# Project history onto the 2D plane and plot as red path
history_projections = []
history_losses = []
for roots in history:
    # Project roots onto dir_x and dir_y
    shifted_roots = roots - stuck_roots
    proj_x = np.dot(shifted_roots, dir_x)
    proj_y = np.dot(shifted_roots, dir_y)
    history_projections.append([proj_x, proj_y])
    history_losses.append(np.log10(compute_loss(roots) + 1e-6)) # Add a small offset to make it visible

history_projections = np.array(history_projections)
history_losses = np.array(history_losses)
ax.plot(history_projections[:, 0], history_projections[:, 1], history_losses, 'r-o', linewidth=2, markersize=4, label='Optimization path')

ax.set_title("Loss Landscape for a degree-5 polynomial's roots (MSE + Coefficients)", fontsize=14)
ax.set_xlabel('Direction X')
ax.set_ylabel('Direction Y')
ax.set_zlabel('Log10(MSE Loss)')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Loss Scale')
ax.legend()

plt.show()