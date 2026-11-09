import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.spatial.distance import cdist
from scipy.special import hankel2

matplotlib.use("QtAgg")


frequency = 10e9
phase_velocity = 1 / np.sqrt(epsilon_0 * mu_0)
wavelength = phase_velocity / frequency
radius = wavelength / 2
k0 = 2.0 * np.pi * frequency / phase_velocity
z0 = np.sqrt(mu_0 / epsilon_0)
e0 = 1.0
e_incident = lambda x: e0 * np.exp(-1j * k0 * x)

number_of_source_points = 100
angles = np.linspace(0, 2.0 * np.pi, number_of_source_points, False)
x_prime = radius * np.cos(angles)
y_prime = radius * np.sin(angles)
source_points = np.stack([x_prime, y_prime], axis=-1)
nx_prime = np.cos(angles)
ny_prime = np.sin(angles)
source_normals = np.stack([nx_prime, ny_prime], axis=-1)
sn = np.sqrt((x_prime[1] - x_prime[0]) ** 2 + (y_prime[1] - y_prime[0]) ** 2)

front_coeff = k0 * sn / 4j
point_diff = source_points[:, None, :] - source_points[None, :, :]
point_diff_norm = np.linalg.norm(point_diff, axis=-1)
hankel_arg = k0 * point_diff_norm

np.fill_diagonal(point_diff_norm, 1.0)  # avoid warning
mom_matrix = (
    front_coeff
    * hankel2(1, k0 * point_diff_norm)
    * (np.sum(source_normals[None, :, :] * point_diff, axis=-1))
    / (point_diff_norm)
)

diagonal_element = -1 / 2
np.fill_diagonal(mom_matrix, diagonal_element)

b = -1 / z0 * e_incident(x_prime)

induced_current = np.linalg.solve(mom_matrix, b)

number_of_observation_points = 500
observation_radius = 4.0 * wavelength
x = np.linspace(-observation_radius, observation_radius, number_of_observation_points)
y = np.linspace(-observation_radius, observation_radius, number_of_observation_points)
xs, ys = np.meshgrid(x, y)
observation_points = np.column_stack([xs.ravel(), ys.ravel()])
r = cdist(observation_points, source_points)
greens_function = 1 / 4j * hankel2(0, k0 * r)
integral = greens_function @ (induced_current * sn)
h_total = 1 / z0 * e_incident(xs) - integral.reshape(xs.shape)

h_scattered = h_total - 1 / z0 * e_incident(xs)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(np.rad2deg(angles), np.abs(induced_current), color="black")
ax[0].scatter(np.rad2deg(angles), np.abs(induced_current), 5, color="black")
mesh = ax[1].pcolormesh(x, y, np.abs(h_total), cmap="hot")
ax[1].set_aspect("equal")
fig.colorbar(mesh)
plt.show()
