from collections import namedtuple

import numpy as np
from pyrsistent import v
import yaml
from numba import jit, njit

from StringUtils import tprint
from calculations import _update_T, _update_T_3D


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

    def check_init(self):
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
        self.check_init()
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
        self.T = T
        self.M = M

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

    def check_init(self):
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

        self.dt = self.timevars.tmax / (self.timevars.nt)  # timestep
        self.dx = self.gridvars.xmax / (self.gridvars.nx)  # x-step
        self.dy = self.gridvars.ymax / (self.gridvars.ny)  # y-step
        self.dz = self.gridvars.zmax / (self.gridvars.nz)  # z-step

        self.check_init()
        print(f"Problem size: {self.gridvars.nx * self.gridvars.ny * self.gridvars.nz}")
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
        x = np.linspace(0, grid.xmax, grid.nx)
        y = np.linspace(0, grid.ymax, grid.ny)
        z = np.linspace(0, grid.zmax, grid.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Preallocate for temperature and mask arrays:
        T = np.zeros((grid.nx, grid.ny, grid.nz, tm.nt))  # temperatures grid
        M = np.zeros((grid.nx, grid.ny, grid.nz, tm.nt))  # mask grid

        # Set up the geometry of the problem with a mask:
        # Top n_atmos cells are void (mask value == 0)
        M0 = np.ones((grid.nx, grid.ny, grid.nz))
        M0[:, :, : grid.n_atmos] = 0
        # Set 'stones' to have a mask value of 2
        M0[:, :, int(grid.nz * ground.h_stone) :] *= 2
        # Assign to first time slice:
        M[:, :, :, 0] = M0

        # Set the first  'plane' of initial conditions:
        # 1D linear temperature profile with depth:
        T0_1d = np.append(
            temp.T_a * np.ones(grid.n_atmos),
            np.linspace(temp.T_g, temp.T_b, grid.nz - 1),
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

    def __init__(self, coeff_filename):
        super().__init__(coeff_filename)
        self.set_init()
        self.build_grid()

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
            self.gridvars.nx,
            self.gridvars.ny,
            self.gridvars.nz,
            self.dt,
            self.tempvars.k_heat,
            self.tempvars.T_a,
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

    def sanitize_boundary(self, last_step):
        """Enforces the fixed temperature boundary condition
        at the base of the active layer in the case that the
        FTCS scheme diffuses into it"""
        # The active layer boundary traverses all columns and rows of
        # the bottom 'sheet'.
        sani_step = last_step.copy()
        sani_step[:, :, -1] = self.tempvars.T_b
        return sani_step

    def timeloop(self):
        """Method which holds the main timeloop."""
        t = self.timevars
        for n in range(t.nt - 1):
            self.last_step = self.T[:, :, :, n]
            self.last_mask = self.M[:, :, :, n]
            next_step = self.update_T_3D()
            self.T[:, :, :, n + 1] = next_step
            self.M[:, :, :, n + 1] = self.update_mask()
            # FIXME: after delta_T from the last time step is calculated:
            # find cells in T that have w > 0 and T < 0
            # check if frozen
            # juggle cells
