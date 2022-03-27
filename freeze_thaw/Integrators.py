from collections import namedtuple

import numpy as np
import yaml
from numba import jit, njit

from StringUtils import tprint
from calculations import (
    _update_T,
    _update_T_3D,
    get_freezing_latent_heat,
    joules_from_delta_T,
)
from tqdm import tqdm
from copy import copy


class Integrator:
    """Integrator class to handle the configuration of the freeze-thaw model.
    Inherit from this base class and supply your own derivative method,
    initialization routines and timeloop."""

    def __init__(self, coeff_filename):
        with open(coeff_filename, "rb") as f:
            config = yaml.safe_load(f)
        self.config = config


class Integrator2D(Integrator):
    """Integrator sub-class to handle the bulk of the 2D freeze-thaw model.
    Inherit from this base class and supply your own derivative method,
    initialization routines and timeloop."""

    def check_timestep_stability(self):
        """Function to check that the stability criterion for a
        FTCS numerical scheme is met, if not, defaults to the smallest
        allowable time-step where the scheme becomes stable."""
        stability_crit = np.min((self.dx, self.dy)) ** 2 / (4 * self.tempvars.k_heat)
        if self.dt > stability_crit:
            print("Warning! Unstable timestep")
            tprint(f"{self.dt=} > {stability_crit=}")
            tprint(f"Defaulting to minimum timestep {stability_crit=}")
            self.dt = stability_crit
        else:
            print("Timestep is stable: ")
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

        # Add in key integration params as attributes:
        self.dt = self.timevars.tmax / (self.timevars.nt)  # timestep
        self.dx = self.gridvars.xmax / (self.gridvars.nx)  # x-step
        self.dy = self.gridvars.ymax / (self.gridvars.ny)  # y-step

        # Depth of model including atmospheric cells:
        self.ny = self.gridvars.ny + self.gridvars.n_atmos
        self.nx = self.gridvars.nx

        # Check the timestep is stable and correct it if not: s
        self.check_timestep_stability()
        tprint(f"{self.nx=}")
        tprint(f"{self.ny=}")
        tprint(f"{self.gridvars.nx=}")
        tprint(f"{self.gridvars.ny=}")
        tprint(f"{self.timevars.nt=} s")
        tprint(f"{self.dx=} m")
        tprint(f"{self.dy=} m")
        tprint(f"{self.dt=} m")

    def build_grid(self):
        """Method to construct a NxM model domain given requested
        parametrs."""

        grid = self.gridvars
        ground = self.groundvars
        temp = self.tempvars
        tm = self.timevars

        # Set up linearly spaced grid:
        x = np.linspace(0, grid.xmax, grid.nx)
        # Make sure to add the extra cells for the atmosphere on top:
        y = np.linspace(0, grid.ymax + grid.n_atmos * self.dy, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Preallocate for temperature and mask arrays:
        T = np.zeros((self.ny, grid.nx, tm.nt))  # temperatures grid
        M = np.zeros((self.ny, grid.nx, tm.nt))  # mask grid

        # Set up the geometry of the problem with a mask:
        # Top n_atmos cells are void (mask value == 0)
        M0 = np.ones((self.ny, grid.nx))
        M0[: grid.n_atmos, :] = 0
        # Set 'stones' to have a mask value of 2
        M0[int(self.ny * ground.h_stone) :, :] *= 2
        # Assign to first time slice:
        M[:, :, 0] = M0

        # Set the first  'plane' of initial conditions:
        # 1D linear temperature profile with depth:
        T0_1d = np.append(
            temp.T_a * np.ones(grid.n_atmos),
            np.linspace(temp.T_g, temp.T_b, grid.ny),
        )
        # Map to 2D and set to first timestep:
        T0 = (T0_1d * np.ones_like(X)).T

        # Boundary conditions on Temperature:
        T[grid.n_atmos :, :] = temp.T_a
        # Bottom row for all time slices is bottom of active layer:
        T0[-1, :] = temp.T_b
        # Assign to first time slice:
        T[:, :, 0] = T0

        # Set attributes:
        self.T = T  # temperature
        self.M = M  # mask

    def __init__(self, coeff_filename):
        super().__init__(coeff_filename)
        self.set_init()
        self.build_grid()

    def update_T(self):
        """Wrapper function for _update_T() function which is
        outside of the Integrator class which uses Numba's no-python
        just-in-time compilation (~20x speedup last I checked compared
        to a serial loop with standard compilation)."""
        next_step = _update_T(
            self.last_step,
            self.last_mask,
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
        # FIXME: function currently only returns a copy()
        next_mask = self.last_mask.copy()
        return next_mask

    def timeloop(self):
        """Method which holds the main timeloop."""
        t = self.timevars
        for n in range(t.nt - 1):
            self.last_step = self.T[:, :, n]
            self.last_mask = self.M[:, :, n]
            self.T[:, :, n + 1] = self.update_T()
            self.M[:, :, n + 1] = self.update_mask()
            # FIXME: after delta_T from the last time step is calculated:
            # find cells in T that have w > 0 and T < 0
            # check if frozen
            # juggle cells


class Integrator3D(Integrator):
    """Integrator class to handle the bulk of the 3D freeze-thaw model.
    Inherit from this base class and supply your own derivative method,
    initialization routines and timeloop."""

    def check_timestep_stability(self):
        """Function to check that the stability criterion for a
        FTCS numerical scheme is met, if not, defaults to the smallest
        allowable time-step where the scheme becomes stable."""
        stability_crit = np.min((self.dx, self.dy, self.dz)) ** 2 / (
            8 * self.tempvars.k_heat
        )
        if self.dt > stability_crit:
            print("Warning! Unstable timestep")
            tprint(f"{self.dt=} > {stability_crit=}")
            tprint(f"Defaulting to minimum timestep {stability_crit=}")
            self.dt = stability_crit
            self.nt = int(np.floor(self.timevars.tmax / self.dt))
            tprint(f"{self.nt=} timesteps to reach {self.timevars.tmax=}")

        else:
            print("Timestep is stable: ")
            tprint(f"{self.dt=} < {stability_crit=}")
            self.nt = self.timevars.nt

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

        self.dt = self.timevars.tmax / (self.timevars.nt)  # timestep
        self.dx = self.gridvars.xmax / (self.gridvars.nx)  # x-step
        self.dy = self.gridvars.ymax / (self.gridvars.ny)  # y-step
        self.dz = self.gridvars.zmax / (self.gridvars.nz)  # z-step

        self.check_timestep_stability()

        # Set spatial step for the entire model including the atmosphere:
        self.nz = self.gridvars.nz + self.gridvars.n_atmos
        self.nx = self.gridvars.nx
        self.ny = self.gridvars.ny

        print(f"Problem size: {self.gridvars.nx * self.gridvars.ny * self.gridvars.nz}")
        tprint(f"{self.nx=}")
        tprint(f"{self.ny=}")
        tprint(f"{self.nz=}")
        tprint(f"{self.dx=}")
        tprint(f"{self.dy=}")
        tprint(f"{self.dz=}")
        print("Integration parameters:")
        tprint(f"{self.timevars.nt=}")
        tprint(f"{self.dt=}\n")

    def build_grid(self):
        """Method to construct a NxM model domain given requested
        parametrs."""
        grid = self.gridvars
        ground = self.groundvars
        temp = self.tempvars
        tm = self.timevars

        # Set up linearly spaced grid:
        x = np.linspace(0, grid.xmax, self.nx)
        y = np.linspace(0, grid.ymax, self.ny)
        z = np.linspace(0, grid.zmax, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Preallocate for temperature and mask arrays:
        T = np.zeros((self.nx, self.ny, self.nz, self.nt))  # temperatures grid
        M = np.zeros_like(T)  # mask grid
        W = np.zeros_like(T)  # water content grid

        # Set up the geometry of the problem with a mask:
        # Top n_atmos cells are void (mask value == 0)
        M0 = np.ones((self.nx, self.ny, self.nz))
        M0[:, :, : grid.n_atmos] = 0
        # Set 'soil' to have a mask value of 2
        M0[:, :, int(self.nz * ground.h_stone) :] *= 2
        # Assign to first time slice:
        M[:, :, :, 0] = M0

        # For the first time slice all soil cells all are assigned
        # an initial water content set by groundvars.
        W0 = np.zeros_like(M0)
        W0[np.where(M0 == 2)] = self.groundvars.w
        W[:, :, :, 0] = W0

        # Set the first  'plane' of initial conditions:
        # 1D linear temperature profile with depth:
        T0_1d = np.append(
            temp.T_a * np.ones(grid.n_atmos),
            np.linspace(temp.T_g, temp.T_b, grid.nz),
        )
        # Map to 2D and set to first timestep:
        T0 = T0_1d * np.ones_like(X)

        # Boundary conditions on Temperature:
        # Top layer n_atmos thick is the atmosphere:
        T[:, :, grid.n_atmos] = temp.T_a
        # Bottom row for all time slices is bottom of active layer:
        T0[:, :, -1] = temp.T_b
        # Assign to first time slice:
        T[:, :, :, 0] = T0

        # Set attributes:
        self.T = T
        self.M = M
        self.X = X
        self.Y = Y
        self.Z = Z
        self.W = W

    def __init__(self, coeff_filename):
        super().__init__(coeff_filename)
        self.set_init()
        self.build_grid()

    def sanitize_boundary(self, last_step):
        """Enforces the fixed temperature boundary condition
        at the base of the active layer in the case that the
        FTCS scheme diffuses into it"""
        # The active layer boundary traverses all columns and rows of
        # the bottom 'sheet'.
        sanitized_step = last_step.copy()
        sanitized_step[:, :, -1] = self.tempvars.T_b
        return sanitized_step

    def update_T_3D(self):
        """Wrapper function for _update_T() function which is
        outside of the Integrator class which uses Numba's no-python
        just-in-time compilation (~20x speedup last I checked compared
        to a serial loop with standard compilation)."""
        next_step = _update_T_3D(
            self.last_step,
            self.last_mask,
            self.dx,
            self.dy,
            self.dz,
            self.nx,
            self.ny,
            self.nz,
            self.dt,
            self.tempvars.k_heat,
            self.tempvars.T_a,
        )
        return self.sanitize_boundary(next_step)

    def update_mask(self):
        """Method which will handle tracking the occupancy of each cell,
        given as:
        - 0: void
        - 1: stonme
        - 2: soil
        - 3: ice
        - 4: void
        """
        # FIXME: function currently only returns a copy()
        next_mask = self.last_mask.copy()
        return next_mask

    def index_sign_change(self, next_step):
        # signum of last temperature time slice
        last_sign = np.sign(self.last_step)
        # signum of next time slice
        next_sign = np.sign(next_step)
        # return as 1 if sign changes from pos to neg and otherwise
        # return zero. works because temperature only decreases with
        # time, i.e. no heating in this specific problem.
        sign_delta = last_sign - next_sign
        return sign_delta

    def timeloop(self):
        """Method which holds the main timeloop."""
        eps = 1e-4
        for n in tqdm(range(self.nt - 1)):
            self.last_step = self.T[:, :, :, n]
            self.last_mask = self.M[:, :, :, n]

            self.W[:, :, :, n + 1] = copy(self.W[:, :, :, n])

            next_step = self.update_T_3D()

            if np.allclose(self.last_step, next_step, rtol=eps):
                print(f"Timeloop converged after {n=} iterations. Exiting.")
                return

            # update temperatures:
            self.T[:, :, :, n + 1] = next_step
            # self.M[:, :, :, n + 1] = self.update_mask()

            # correct for latent heat release:
            sign_delta = self.index_sign_change(next_step)
            if np.any(sign_delta):
                indices = np.argwhere(sign_delta)

                for index in indices:
                    i, j, k = index
                    # the value in next_cell is the portion
                    # of the total delta_T from last --> next steps
                    # that is below zero (this <0 temperature drop is
                    # what does the work.)
                    next_cell = next_step[i, j, k]

                    last_cell = self.last_step[i, j, k]
                    water_content = self.W[i, j, k, n]
                    water_mass = water_content * self.dx * self.dy * self.dz * 0.5

                    total_latent_heat = get_freezing_latent_heat(water_mass)
                    print(f"{total_latent_heat=}")
                    joules_released = joules_from_delta_T(water_mass, next_cell)
                    print(f"{joules_released=}")
                    proportion_frozen = joules_released / total_latent_heat

                    # update water content:
                    next_water = water_content - water_content * (1 - proportion_frozen)

                    # update temperature grid:
                    self.T[i, j, k, n + 1] = 0

                    if next_water < 0:
                        print("Freezing front overshot")
                        # correct the temperature grid:
                        # calculate mass of water overshot
                        # find how many joules it would have taken to freeze that mass
                        # find what delta_T arises from re-adding those many joules
                        # update the temperature grid with delta_T
                    self.W[i, j, k, n + 1] = next_water

            # FIXME: after delta_T from the last time step is calculated:
            # find cells in T that have w > 0 and T < 0
            # check if frozen
            # juggle cells
            # if any in T where M > 0 has sign change from last --> next:
