#%%
import numpy as np
import matplotlib.pyplot as plt

from numba import njit, jit


@njit
def _update_T_3D(
    last_step,
    last_mask,
    dx,
    dy,
    dz,
    nx,
    ny,
    nz,
    dt,
    k_heat,
):
    """Forward time centered space numerical scheme to calculate
    3D heat diffusion in a plane:"""

    # Factors for explicit scheme:
    diagx = -2 * (dx ** 2) / (2 * k_heat * dt)
    diagy = -2 * (dy ** 2) / (2 * k_heat * dt)
    diagz = -2 * (dz ** 2) / (2 * k_heat * dt)
    weightx = k_heat * dt / (dx ** 2)
    weighty = k_heat * dt / (dy ** 2)
    weightz = k_heat * dt / (dz ** 2)
    # copy the time step specifically to pull out the top and bottom
    # boundary conditions (FIXME)
    next_step = last_step.copy()

    for i in np.arange(0, nx):
        for j in np.arange(0, ny):
            # always skip the last layer of z-cells because the lower boundary is fixed...
            for k in np.arange(0, nz - 1):

                # Check if the [i, j]'th cell is atmospheric:
                if last_mask[i, j, k] == 0:
                    # If cell is air, T[i,j] should equal T_a and
                    # we can do nothing here. NOTE: a sanitization step
                    # is necessary higher in this stack to check that atmospheric
                    # cells aren't accumulating heat.
                    continue

                # Handle the periodic boundary conditions inelegantly for now:
                if i == nx - 1 and j < ny - 1:
                    # For the case where the loop hits the i  = (nx - 1)'th element,
                    # there is no T[i+1, j, k]'th element because MAX(i) == nx - 1 and
                    # i + 1 when i == nx implies an element which is not indexable but
                    # physically is the i == 0th element:
                    next_step[i, j, k] = (
                        weightx
                        * (
                            last_step[i - 1, j, k]
                            + last_step[0, j, k]  # !
                            + last_step[i, j, k] * diagx
                        )
                        + weighty
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, j + 1, k]
                            + last_step[i, j, k] * diagy
                        )
                        + weightz
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            + last_step[i, j, k] * diagz
                        )
                    )
                elif j == ny - 1 and i < nx - 1:
                    next_step[i, j, k] = (
                        weightx
                        * (
                            last_step[i - 1, j, k]
                            + last_step[i + 1, j, k]
                            + last_step[i, j, k] * diagx
                        )
                        + weighty
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, 0, k]
                            + last_step[i, j, k] * diagy
                        )
                        + weightz
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            + last_step[i, j, k] * diagz
                        )
                    )
                elif j == ny - 1 and i == nx - 1:
                    next_step[i, j, k] = (
                        weightx
                        * (
                            last_step[i - 1, j, k]
                            + last_step[0, j, k]
                            + last_step[i, j, k] * diagx
                        )
                        + weighty
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, 0, k]
                            + last_step[i, j, k] * diagy
                        )
                        + weightz
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            + last_step[i, j, k] * diagz
                        )
                    )
                else:
                    next_step[i, j, k] = (
                        weightx
                        * (
                            last_step[i - 1, j, k]
                            + last_step[i + 1, j, k]
                            + last_step[i, j, k] * diagx
                        )
                        + weighty
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, j + 1, k]
                            + last_step[i, j, k] * diagy
                        )
                        + weightz
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            + last_step[i, j, k] * diagz
                        )
                    )
    return next_step


@njit
def _update_T(
    last_step,
    last_mask,
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
    # start the loop from the top of the simulation but continue
    # if the cell is masked as atmospheric (or void)
    # always skip the last step because the lower boundary is fixed...
    for i in np.arange(0, ny - 1):
        for j in np.arange(0, nx):
            # Check if the [i, j]'th cell is atmospheric:
            if last_mask[i, j] == 0:
                # If cell is air, T[i,j] should equal T_a and
                # we can do nothing here. NOTE: a sanitization step
                # is necessary higher in this stack to check that atmospheric
                # cells aren't accumulating heat.
                continue
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


class GridCell:
    def __init__(self):
        self.void = False
        self.frozen = False
        self.n_stones = []
        self.n_soil = []
        self.w = []
        self.temp = []
        self.last_temp = []

    def set_domain(self):
        if self.n_stones == 1:
            if self.n_soil == 1:
                self.domain = "soil"
            else:
                self.domain = "stone"

    def get_latent_heat(self, delta_T):
        """At zero degrees, some delta_T is incurred as the heat
        equation is stepped through time. Convert delta_T to
        the equivalent energy input based on the specific heat
        capacity of the cell.
        """
        pass

    def check_frozen(self, delta_T):
        """Cell is considered 'frozen' when
        - w = 0: water content is zero
        - T <= 0: temperature is freezing
        """
        # stone cells can't freeze:
        if self.domain == "stone":
            return

        if self.last_temp + delta_T <= 0:
            if self.w == 0:
                self.frozen = True
            else:
                joules = self.get_latent_heat(delta_T)


# #%% Spatial parameters ::
# # make a 2d grid
# dx = 0.1  # grid spacing
# L = 10  # 20m in Kessler
# N = int(L // dx)  # number of grid cells per dimension
# x = np.linspace(0, L, N)  # 1d axis
# X, Y = np.meshgrid(x, x)  # 2d coordinates


#%%# %%
