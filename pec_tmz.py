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

number_of_source_points = 50
angles = np.linspace(0, 2.0 * np.pi, number_of_source_points, False)
rp = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)

integral_segment_length = radius * (angles[1] - angles[0])

coefficient = k0 * z0 * integral_segment_length / 4
rp_rp_difference_magnitude = np.linalg.norm(rp[:, None, :] - rp[None, :, :], axis=-1)

np.fill_diagonal(rp_rp_difference_magnitude, 1.0)  # avoid warning when hankel2 called
mom_matrix = coefficient * hankel2(0, k0 * rp_rp_difference_magnitude)

gamma = 1.781  # from textbook
diagonal_element = coefficient * (
    1.0
    - 1j * 2 / np.pi * np.log(k0 * gamma * integral_segment_length / (4 * np.exp(1)))
)
np.fill_diagonal(mom_matrix, diagonal_element)

b = e_incident(radius * np.cos(angles))
induced_current = np.linalg.solve(mom_matrix, b)

# calculate fields at observation points
number_of_observation_points = 500
observation_radius = 4.0 * wavelength
x = np.linspace(-observation_radius, observation_radius, number_of_observation_points)
x, y = np.meshgrid(x, x)
r = np.column_stack([x.ravel(), y.ravel()])

r_rp_difference_magnitude = np.linalg.norm(r[:, None, :] - rp[None, :, :], axis=-1)
greens_function = 1 / 4j * hankel2(0, k0 * r_rp_difference_magnitude)
integral = greens_function @ (induced_current * integral_segment_length)
e_total = e_incident(x) - 1j * k0 * z0 * integral.reshape(x.shape)
e_scattered = e_total - e_incident(x)

# plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(np.rad2deg(angles), np.abs(induced_current), color="black")
ax[0].scatter(np.rad2deg(angles), np.abs(induced_current), 5, color="black")
mesh = ax[1].pcolormesh(x, y, np.abs(e_total), cmap="hot")
ax[1].set_aspect("equal")
fig.colorbar(mesh)
plt.show()
