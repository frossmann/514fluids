#%%
import numpy as np
import matplotlib.pyplot as plt

from numba import njit, jit


@njit
def _update_T_3D(last_step, last_mask, dx, dy, dz, nx, ny, nz, dt, k_heat, T_a):
    """Forward time centered space numerical scheme to calculate
    3D heat diffusion in a plane:"""

    # Factors for explicit scheme:
    weight_x = k_heat * dt / (dx ** 2)
    weight_y = k_heat * dt / (dy ** 2)
    weight_z = k_heat * dt / (dz ** 2)
    # copy the time step specifically to pull out the top and bottom
    # boundary conditions (FIXME)
    next_step = np.zeros_like(last_step)

    for i in np.arange(0, nx):
        for j in np.arange(0, ny):
            # NOTE:
            # - skip k = 0 because this is the ideally atmosphere AND
            # k-1 --> k == -1 which samples the bottom of the active layer.
            # - skip k == nz because it's explicitly the bottom of "".
            for k in np.arange(0, nz - 1):

                # Check if the [i, j]'th cell is atmospheric:
                if last_mask[i, j, k] == 0:
                    # If cell is air, T[i,j] should equal T_a and
                    # we can do nothing here. NOTE: a sanitization step
                    # is necessary higher in this stack to check that atmospheric
                    # cells aren't accumulating heat: until then, declare it explicitly:
                    next_step[i, j, k] = T_a
                    continue

                # Handle the periodic boundary conditions inelegantly for now:
                if i == nx - 1 and j < ny - 1:
                    # For the case where the loop hits the i  = (nx - 1)'th element,
                    # there is no T[i+1, j, k]'th element because MAX(i) == nx - 1 and
                    # i + 1 when i == nx implies an element which is not indexable but
                    # physically is the i == 0th element:
                    next_step[i, j, k] = last_step[i, j, k] + (
                        weight_x
                        * (
                            last_step[i - 1, j, k]
                            + last_step[0, j, k]  # !
                            - 2 * last_step[i, j, k]
                        )
                        + weight_y
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, j + 1, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_z
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            - 2 * last_step[i, j, k]
                        )
                    )
                elif j == ny - 1 and i < nx - 1:
                    next_step[i, j, k] = last_step[i, j, k] + (
                        weight_x
                        * (
                            last_step[i - 1, j, k]
                            + last_step[i + 1, j, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_y
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, 0, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_z
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            - 2 * last_step[i, j, k]
                        )
                    )
                elif j == ny - 1 and i == nx - 1:
                    next_step[i, j, k] = last_step[i, j, k] + (
                        weight_x
                        * (
                            last_step[i - 1, j, k]
                            + last_step[0, j, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_y
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, 0, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_z
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            - 2 * last_step[i, j, k]
                        )
                    )
                else:
                    next_step[i, j, k] = last_step[i, j, k] + (
                        weight_x
                        * (
                            last_step[i - 1, j, k]
                            + last_step[i + 1, j, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_y
                        * (
                            last_step[i, j - 1, k]
                            + last_step[i, j + 1, k]
                            - 2 * last_step[i, j, k]
                        )
                        + weight_z
                        * (
                            last_step[i, j, k - 1]
                            + last_step[i, j, k + 1]
                            - 2 * last_step[i, j, k]
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


def get_freezing_latent_heat(mass_h20):
    """Returns energy in the form of latent heat that would
    be released by freezing `mass_h20` of water."""
    heat_of_fusion = 333.55 * 1000  # J/g * 1000g/1kg  [J/kg]
    return mass_h20 * heat_of_fusion


def joules_from_delta_T(mass_h20, delta_temp):
    """Returns joules released that would theoretically
    be released by dropping a `mass_h20`
    by `delta-temp` degrees"""
    delta_temp = delta_temp
    specific_heat_h20 = 4182  # J/(kgC)
    return -specific_heat_h20 * mass_h20 * delta_temp


def lens_creation():
    pass


def find_surf_path():
    pass


def push_to_surface():
    pass


def push_to_void():
    pass


def delta_T_from_joules(mass_h20, joules):
    """Returns temperature delta incurred from that removing
    quantity `joules` from `mass_h20` kg of liquid water, in degC"""
    specific_heat_h20 = 4182  # J/(kgC)
    return joules / (mass_h20 * specific_heat_h20)


# %%
import numpy as np
import matplotlib.pyplot as plt

def diff_1d_analytic(time, kappa = 0.000001):
    """Analytical solution for 1D diffusion equation
    evaluated at time `time` over the spatial domain 
    of the freeze-thaw model and the same diffusion constant"""

    def coeff(n):
        """Coefficient on the fourier terms"""
        bn = 20 * (-np.sin(n * np.pi) + np.pi * n) / (np.pi ** 2 * n ** 2)
        return bn


    def fourier_term(xline, time, L, kappa, n):
        """Returns the n-th term of the fourier series from the
        analytical solution at time `time`"""
        sol_n = (
            coeff(n)
            * np.exp((-kappa * (n * np.pi / L) ** 2) * time)
            * np.sin(n * np.pi * xline / L)
        )
        return sol_n

    def fourier_sum(N, xline, time, L, kappa):
        """Returns the sum of the first N fourier terms 
        in the analytical solution at time `time`"""
        the_sum = np.zeros_like(xline)
        for n in np.arange(1, N+1, 1):
            the_sum += fourier_term(xline, time, L, kappa, n)
        return the_sum

    # problem parameters
    # boundary conditions   
    u0 = -5  # surface
    u1 = 0  # base
    L = 1  # domain depth, m
    dx = 0.05  # cell height, m
    

    # profile line: 
    xline = np.arange(0, L + dx, dx)
    
    # steady state solution: 
    u_ss = u0 + ((u1 - u0) / L) * xline
    N = 1000  # number of fourier terms
    sol_n = u_ss + fourier_sum(N, xline, time, L, kappa)
    return sol_n
