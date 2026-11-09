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
h0 = 1.0 / z0
h_incident = lambda x: h0 * np.exp(-1j * k0 * x)

number_of_source_points = 300
angles = np.linspace(0, 2.0 * np.pi, number_of_source_points, False)
rp = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
unp = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

integral_segment_length = radius * (angles[1] - angles[0])

rp_rp_difference = rp[:, None, :] - rp[None, :, :]
rp_rp_difference_magnitude = np.linalg.norm(rp_rp_difference, axis=-1)

mom_matrix = (
    (k0 * integral_segment_length / 4j)
    * hankel2(1, k0 * rp_rp_difference_magnitude)
    * (1 / rp_rp_difference_magnitude)
    * np.sum(unp[None, :, :] * rp_rp_difference, axis=-1)
)

diagonal_element = -0.5
np.fill_diagonal(mom_matrix, diagonal_element)

b = h_incident(radius * np.cos(angles))
induced_current = np.linalg.solve(mom_matrix, b)

# calculate fields at observation points
number_of_observation_points = 500
observation_radius = 4.0 * wavelength
x = np.linspace(-observation_radius, observation_radius, number_of_observation_points)
x, y = np.meshgrid(x, x)
r = np.column_stack([x.ravel(), y.ravel()])
r_rp_difference = r[:, None, :] - rp[None, :, :]
r_rp_difference_magnitude = np.linalg.norm(r_rp_difference, axis=-1)
greens_function = (
    (k0 / 4j)
    * hankel2(1, k0 * r_rp_difference_magnitude)
    * (1 / r_rp_difference_magnitude)
    * np.sum(unp[None, :, :] * r_rp_difference, axis=-1)
)
integral = greens_function @ (induced_current * integral_segment_length)
h_total = h_incident(x) - integral.reshape(x.shape)
h_scattered = h_total - h_incident(x)

# plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(np.rad2deg(angles), np.abs(induced_current), color="black")
ax[0].scatter(np.rad2deg(angles), np.abs(induced_current), 5, color="black")
mesh = ax[1].pcolormesh(x, y, np.abs(h_total), cmap="hot")
ax[1].set_aspect("equal")
fig.colorbar(mesh)
plt.show()
