from collections import namedtuple

import numpy as np
import yaml
from numba import jit, njit

from StringUtils import tprint


@njit
def _update_T(
    last_step,
    dx,
    dy,
    nx,
    ny,
    dt,
    k_heat,
):
    """Forward time centered space numerical scheme to calculate
    2D heat diffusion in a plane:"""
    # copy the time step specifically to pull out the top and bottom
    # boundary conditions (FIXME)
    next_step = last_step.copy()
    for i in np.arange(1, ny - 1):
        for j in np.arange(0, nx):
            # Set up periodic boundary conditions (note we don't need to do this if
            # j == 0 because then j-1 == -1 which python indexes as the last element
            # so wrapping is already taken care of )
            if j == nx - 1:
                next_step[i, j] = (
                    last_step[i, j]
                    + (dt * k_heat / (dx ** 2))
                    * (last_step[i - 1, j] - 2 * last_step[i, j] + last_step[i + 1, j])
                    + (dt * k_heat / (dy ** 2))
                    * (last_step[i, j - 1] - 2 * last_step[i, j] + last_step[i, 0])
                )
            # otherwise just step through time:
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
    """Integrator class to handle the bulk of the freeze-thaw model"""

    def check_init(self):
        """Function to check that the stability criterion for a
        FTCS numerical scheme is met, if not, defaults to the smallest
        allowable time-step where the scheme becomes stable."""
        stability_crit = np.min((self.dx, self.dy)) ** 2 / (4 * self.tempvars.k_heat)
        if self.dt > stability_crit:
            tprint("Warning! Unstable timestep")
            tprint(f"{self.dt=} > {stability_crit=}")
            tprint(f"Defaulting to minimum timestep {stability_crit=}")
            self.dt = stability_crit
        else:
            tprint("Timestep is stable: ")
            tprint(f"{self.dt=} < {stability_crit=}")

    def set_init(self):
        """Method to set up the physical parameters of the problem, integration
        parameters and others."""
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

        tprint(f"{self.gridvars.nx=}")
        tprint(f"{self.gridvars.ny=}")
        tprint(f"{self.timevars.nt=}")
        tprint(f"{self.dx=}")
        tprint(f"{self.dy=}")
        tprint(f"{self.dt=}")

    def build_grid(self):
        """Method to construct a NxM model domain given requested
        parametrs."""
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
        T = np.zeros((grid.nx, grid.ny, tm.nt))  # temperatures grid
        M = np.ones((grid.nx, grid.ny, tm.nt))  # mask grid
        M[: grid.n_atmos, :, :] = np.zeros(
            (grid.nx, grid.n_atmos)
        )  # uppermost n_atmos cells are 'empty'
        M[int(grid.ny * ground.h_stone) :, :, :] *= 2

        # Set the first  'plane' of initial conditions:
        # 1D linear temperature profile with depth:
        T0_1d = np.append(
            temp.T_a * np.ones(grid.n_atmos),
            np.linspace(temp.T_g, temp.T_b, grid.nx - 1),
        )
        # Map to 2D and set to first timestep:
        T[:, :, 0] = (T0_1d * np.ones_like(X)).T

        # Boundary conditions on Temperature:
        # Bottom row for all time slices is bottom of active layer:
        T[-1, :, :] = temp.T_b * np.ones((grid.nx, tm.nt))

        # # Top row for all time slices is atmospheric temperature:
        for n in range(tm.nt):
            T[0, :, n] = temp.T_a * np.ones(
                grid.nx
            )  # FIXME: atmospheric temperature varies

        self.T = T
        self.M = M

    def __init__(self, coeff_filename):
        with open(coeff_filename, "rb") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.set_init()
        self.build_grid()

    def update_T(self):
        """Wrapper function for _update_T() function which is
        outside of the Integrator class which uses Numba's no-python
        just-in-time compilation (~20x speedup last I checked compared
        to a serial loop with standard compilation)."""
        next_step = _update_T(
            self.last_step,
            self.dx,
            self.dy,
            self.gridvars.nx,
            self.gridvars.ny,
            self.dt,
            self.tempvars.k_heat,
        )
        return next_step

    def update_mask(self):
        """Method which will handle tracking the occupancy of each cell,
        given as:
        - 0: void
        - 1: stonme
        - 2: soil
        - 3: ice
        - 4: void
        """
        next_mask = np.ones_like(self.M[:, :, 0])
        next_mask[self.last_mask == 0] = self.T_a

    def timeloop(self):
        """Method which holds the main timeloop."""
        t = self.timevars
        for n in range(t.nt - 1):
            self.last_step = self.T[:, :, n]
            self.T[:, :, n + 1] = self.update_T()
            # FIXME: after delta_T from the last time step is calculated:
            # find cells in T that have w > 0 and T < 0
            # check if frozen
            # juggle cells
