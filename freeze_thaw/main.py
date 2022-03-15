#%%
import time
import timeit
from collections import namedtuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numba import jit, njit

from Integrators import Integrator
from PlotUtils import animate
from StringUtils import tprint

if __name__ == "__main__":
    sim = Integrator("/Users/francis/repos/514fluids/freeze_thaw/uservars.yaml")
    sim.build_grid()
    sim.timeloop()

    ani = animate(sim)


#%%
# # Integration parameters:
# nt = 5000  # number of time steps
# tmax = 1  # max. integration time
# nx = 25  # number of x-steps
# xmax = 1  # upper x-limit
# ny = nx  # number of y-steps
# ymax = 1  # upper y-limit
# # k_heat = 0.2     # diffusion constant
# dt = tmax / (nt - 1)  # timestep
# dx = xmax / (nx - 1)  # x-step
# dy = ymax / (ny - 1)  # y-step

# # Temperature:
# T_a = -5 * np.ones(nt)  # degC
# T_b = 0  # degC
# T_g = 5  # degC
# k_heat = 0.2  # 1e-06  # m2s-1
# k_surf = 5e-3  # m2yr-1


# stability_crit = dx ** 2 / (4 * k_heat)
# print(f"Timestep {dt=} should be less than {stability_crit=}")
# if dt > stability_crit:
#     print("Warning! Unstable timestep")
#     print(f"{dt=} > {stability_crit=}")
#     print(f"Defaulting to minimum timestep {stability_crit=}")
# else:
#     print(f"{dt=} < {stability_crit=}")

# epsilon = 1e-3


# # Heave distance:
# d_s = 0.6
# d_v = 0.6

# # Layer thicknesses:
# h_stone = 0.6
# h_soil = 0.4

# # Soil parameters:
# w = 0.1  # water content
# C = 0.05  # compressibility


# # Set up linearly spaced grid:
# x = np.linspace(0, xmax, nx)
# y = np.linspace(0, ymax, ny)
# X, Y = np.meshgrid(x, y, indexing="xy")

# # Initialize particles:
# to_interface = int(nx * h_stone)
# P = np.ones_like(X)
# P[0:to_interface] = 0
# P[to_interface:] = 1

# #% Initialize temperatures with initial conditions:
# # Temperature grid:
# T = np.zeros((nx, ny, nt))

# T0_1d = np.append(T_a[0], np.linspace(T_g, T_b, nx - 1))
# T0_2d = (T0_1d * np.ones_like(X)).T
# T[:, :, 0] = T0_2d

# # Boundary conditions on Temperature:
# # Bottom row for all time slices is bottom of active layer:
# T[-1, :, :] = T_b * np.ones((nx, nt))

# # # Top row for all time slices is atmospheric temperature:
# for n in range(nt):
#     T[0, :, n] = T_a[n] * np.ones(nx)  # FIXME: atmospheric temperature varies


# # #%% check your work
# # fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 10))
# # ax[0].imshow(T[:,:,0])
# # ax[0].set_title("Initial temperature grid")
# # ax[1].imshow(P)
# # ax[1].set_title("Initial particle grid")

# # ax[1].set_xlabel("Width (grid units)")
# # ax[0].set_ylabel("Depth (grid units)")

# #%%

# # #
# # # Loop for explicit FTCS scheme:
# # @njit
# # def FTCS_2D(nt, nx, ny, dt, dx, dy, k_heat, T):
# #     for n in range(nt - 1):
# #         for i in np.arange(1, nx - 1):
# #             for j in np.arange(0, ny):
# #                 if j == ny - 1:
# #                     T[i, j, n + 1] = (
# #                         T[i, j, n]
# #                         + (dt * k_heat / (dx ** 2))
# #                         * (T[i - 1, j, n] - 2 * T[i, j, n] + T[i + 1, j, n])
# #                         + (dt * k_heat / (dy ** 2))
# #                         * (T[i, j - 1, n] - 2 * T[i, j, n] + T[i, 0, n])
# #                     )
# #                 else:
# #                     T[i, j, n + 1] = (
# #                         T[i, j, n]
# #                         + (dt * k_heat / (dx ** 2))
# #                         * (T[i - 1, j, n] - 2 * T[i, j, n] + T[i + 1, j, n])
# #                         + (dt * k_heat / (dy ** 2))
# #                         * (T[i, j - 1, n] - 2 * T[i, j, n] + T[i, j + 1, n])
# #                     )
# #         # if np.allclose(T[:,:,n], T[:,:,n-1], rtol=epsilon):
# #         #     print(f'Solution converged after {n=} timesteps')
# #         #     break
# #     return T, n


# # t0 = time.perf_counter()
# # T, n = FTCS_2D(nt, nx, ny, dt, dx, dy, k_heat, T)
# # print(f"Elapsed time: {time.perf_counter() - t0}")
# # # %timeit FTCS(nt, nx, ny, dt, dx, dy, k_heat, epsilon, T)

# # #%%
# # # Plots:
# # time_idx = [0, int(nt / 10), int(nt / 2), int(n)]
# # clim = [-5, 5]
# # for ij in range(len(time_idx)):
# #     fig = plt.figure(ij, figsize=(11, 7), dpi=100)
# #     plt.gca().invert_yaxis()
# #     ax = fig.gca()
# #     ax.set_xlabel("Length")
# #     ax.set_ylabel("Depth")
# #     # surf=ax.contourf(X,Y,T[:,:,time_idx[ij]])
# #     surf = ax.imshow(T[:, :, time_idx[ij]], vmin=clim[0], vmax=clim[1])
# #     # ax.axis('equal')
# #     theTitle = "fig. {}: 2D Heat Diffusion from a Line Source, t={}".format(
# #         ij + 1, time_idx[ij]
# #     )
# #     ax.set_title(theTitle)
# #     cbar = plt.colorbar(surf)
# #     cbar.set_label("Temperature")
# #     plt.show()
