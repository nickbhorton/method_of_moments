import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2

frequency = 10e9
phase_velocity_0 = 1 / np.sqrt(epsilon_0 * mu_0)
phase_velocity_i = 1 / np.sqrt(9.0 * epsilon_0 * mu_0)
wavelength = phase_velocity_0 / frequency
radius = wavelength / 2
k0 = 2.0 * np.pi * frequency / phase_velocity_0
ki = 2.0 * np.pi * frequency / phase_velocity_i
z0 = np.sqrt(mu_0 / epsilon_0)
incident = lambda x: np.exp(-1j * k0 * x)

N = 100
angles = np.linspace(0, 2.0 * np.pi, N, False)
sn = radius * (angles[1] - angles[0])
rp = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
unp = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
rprp_diff = rp[:, None, :] - rp[None, :, :]
rprp_diff_mag = np.linalg.norm(rprp_diff, axis=-1)
np.fill_diagonal(rprp_diff_mag, 1.0)  # avoid warning in hankel2

dG0 = (
    k0
    / 4j
    * hankel2(1, k0 * rprp_diff_mag)
    * np.sum(unp[None, :, :] * rprp_diff, axis=-1)
    / rprp_diff_mag
)
G0 = 1 / 4j * hankel2(0, k0 * rprp_diff_mag)
np.fill_diagonal(dG0, -1 / (4 * np.pi * radius))
np.fill_diagonal(
    G0, 1 / 4j * (1 - 2j / np.pi * np.log(k0 * 1.781 * sn / (4 * np.exp(1))))
)

A0 = sn * dG0 - 0.5 * np.identity(N)
B0 = -1j * k0 * z0 * sn * G0
b0 = -incident(radius * np.cos(angles))

dGi = (
    ki
    / 4j
    * hankel2(1, ki * rprp_diff_mag)
    * np.sum(unp[None, :, :] * rprp_diff, axis=-1)
    / rprp_diff_mag
)
Gi = 1 / 4j * hankel2(0, ki * rprp_diff_mag)
np.fill_diagonal(dGi, -1 / (4 * np.pi * radius))
np.fill_diagonal(
    Gi, 1 / 4j * (1 - 2j / np.pi * np.log(ki * 1.781 * sn / (4 * np.exp(1))))
)

Ai = sn * dGi + 0.5 * np.identity(N)
Bi = -1j * k0 * z0 * sn * Gi
bi = np.zeros_like(b0)

mom_matrix = np.block([[A0, B0], [Ai, Bi]])
b = np.block([b0, bi])

x = np.linalg.solve(mom_matrix, b)
ans = x[0 : len(x) // 2]
bns = x[len(x) // 2 :]

number_of_observation_points = 500
observation_radius = 2.0 * wavelength
x = np.linspace(-observation_radius, observation_radius, number_of_observation_points)
x, y = np.meshgrid(x, x)

r = np.column_stack([x.ravel(), y.ravel()])
r_rp_diff = r[:, None, :] - rp[None, :, :]
r_rp_diff_mag = np.linalg.norm(r_rp_diff, axis=-1)

dG0_r_rp = (
    (k0 / 4j)
    * hankel2(1, k0 * r_rp_diff_mag)
    * np.sum(unp[None, :, :] * r_rp_diff, axis=-1)
    / r_rp_diff_mag
)
a0_sum = dG0_r_rp @ (sn * ans)

G0_r_rp = 1 / 4j * hankel2(0, k0 * r_rp_diff_mag)
b0_sum = G0_r_rp @ (sn * bns)

dGi_r_rp = (
    (ki / 4j)
    * hankel2(1, ki * r_rp_diff_mag)
    * np.sum(unp[None, :, :] * r_rp_diff, axis=-1)
    / r_rp_diff_mag
)
ai_sum = dGi_r_rp @ (sn * ans)

Gi_r_rp = 1 / 4j * hankel2(0, ki * r_rp_diff_mag)
bi_sum = Gi_r_rp @ (sn * bns)

Ez0 = (
    2 * incident(x)
    + 2.0 * a0_sum.reshape(x.shape)
    - 2j * k0 * z0 * b0_sum.reshape(x.shape)
)

Ezi = -2.0 * ai_sum.reshape(x.shape) + 2j * k0 * z0 * bi_sum.reshape(x.shape)

rho = np.sqrt(x**2 + y**2)
near = np.abs(rho - radius) < 0.1 * sn
Ez = np.where(rho <= radius, Ezi, Ez0)
Ez = np.where(near, np.nan, Ez)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(x, y, np.real(Ez), cmap="bwr")
fig.colorbar(mesh)
plt.show()
