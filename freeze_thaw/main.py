#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
import timeit
import time
import yaml
from collections import namedtuple

# from StringUtils import print


@njit
def _update_T(last_step, dx, dy, nx, ny, dt, k_heat):
    next_step = np.zeros_like(last_step)
    for i in np.arange(1, ny - 1):
        for j in np.arange(0, nx):
            if j == nx - 1:
                next_step[i, j] = (
                    last_step[i, j]
                    + (dt * k_heat / (dx ** 2))
                    * (last_step[i - 1, j] - 2 * last_step[i, j] + last_step[i + 1, j])
                    + (dt * k_heat / (dy ** 2))
                    * (last_step[i, j - 1] - 2 * last_step[i, j] + last_step[i, 0])
                )
            else:
                next_step[i, j] = (
                    last_step[i, j]
                    + (dt * k_heat / (dx ** 2))
                    * (last_step[i - 1, j] - 2 * last_step[i, j] + last_step[i + 1, j])
                    + (dt * k_heat / (dy ** 2))
                    * (last_step[i, j - 1] - 2 * last_step[i, j] + last_step[i, j + 1])
                )
    return next_step


class Integrator:
    def check_init(self):
        stability_crit = np.min((self.dx, self.dy)) ** 2 / (4 * self.tempvars.k_heat)
        # if self.dt > stability_crit:
        #     print("Warning! Unstable timestep")
        #     print(f"{self.dt=} > {stability_crit=}")
        #     print(f"Defaulting to minimum timestep {stability_crit=}")
        # else:
        #     print("Timestep is stable: ")
        #     print(f"{self.dt=} < {stability_crit=}")

    def set_init(self):
        timevars = namedtuple("timevars", self.config["timevars"].keys())
        self.timevars = timevars(**self.config["timevars"])
        gridvars = namedtuple("gridvars", self.config["gridvars"].keys())
        self.gridvars = gridvars(**self.config["gridvars"])
        tempvars = namedtuple("tempvars", self.config["tempvars"].keys())
        self.tempvars = tempvars(**self.config["tempvars"])
        groundvars = namedtuple("groundvars", self.config["groundvars"].keys())
        self.groundvars = groundvars(**self.config["groundvars"])

        self.dt = self.timevars.tmax / (self.timevars.nt - 1)  # timestep
        self.dx = self.gridvars.xmax / (self.gridvars.nx - 1)  # x-step
        self.dy = self.gridvars.ymax / (self.gridvars.ny - 1)  # y-step

        self.check_init()

    def build_grid(self):
        grid = self.gridvars
        ground = self.groundvars
        temp = self.tempvars
        tm = self.timevars

        # Set up linearly spaced grid:
        x = np.linspace(0, grid.xmax, grid.nx)
        y = np.linspace(0, grid.ymax, grid.ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        #% Initialize temperatures with initial conditions:
        # Temperature grid:
        T = np.zeros((grid.nx, grid.ny, tm.nt))

        T0_1d = np.append(temp.T_a, np.linspace(temp.T_g, temp.T_b, grid.nx - 1))
        T0_2d = (T0_1d * np.ones_like(X)).T
        T[:, :, 0] = T0_2d

        # Boundary conditions on Temperature:
        # Bottom row for all time slices is bottom of active layer:
        T[-1, :, :] = temp.T_b * np.ones((grid.nx, tm.nt))

        # # Top row for all time slices is atmospheric temperature:
        for n in range(tm.nt):
            T[0, :, n] = temp.T_a * np.ones(
                grid.nx
            )  # FIXME: atmospheric temperature varies

        self.T = T

    def __init__(self, coeff_filename):
        with open(coeff_filename, "rb") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.set_init()
        self.build_grid()

    def update_T(self):
        return _update_T(
            self.last_step,
            self.dx,
            self.dy,
            self.nx,
            self.ny,
            self.dt,
            self.tempvars.k_heat,
        )

    def timeloop(self):
        t = self.timevars
        for n in range(t.nt - 1):
            self.last_step = self.T[:, :, n]
            self.T[:, :, n + 1] = self.update_T()


if __name__ == "__main__":
    tst = Integrator("uservars.yaml")
    tst.build_grid()
    tst.timeloop()
    nt = tst.timevars.nt
    time_idx = [0, int(nt / 10), int(nt / 2), int(n)]
    clim = [-5, 5]
    for ij in range(len(time_idx)):
        fig = plt.figure(ij, figsize=(11, 7), dpi=100)
        plt.gca().invert_yaxis()
        ax = fig.gca()
        ax.set_xlabel("Length")
        ax.set_ylabel("Depth")
        # surf=ax.contourf(X,Y,T[:,:,time_idx[ij]])
        surf = ax.imshow(tst.T[:, :, time_idx[ij]], vmin=clim[0], vmax=clim[1])
        # ax.axis('equal')
        theTitle = "fig. {}: 2D Heat Diffusion from a Line Source, t={}".format(
            ij + 1, time_idx[ij]
        )
        ax.set_title(theTitle)
        cbar = plt.colorbar(surf)
        cbar.set_label("Temperature")
        plt.show()


# # import calculations as calc


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

# # # %%

# %%
