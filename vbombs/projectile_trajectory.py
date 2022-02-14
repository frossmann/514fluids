#%%
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib auto


def analytical(vel, C_d=1, diam=1, rho_f=1.3, rho_p=2700, n=0):
    """Returns analytical solution to the problem"""
    delta_rho = rho_f - rho_p
    g = -9.81  # gravity
    k = (C_d * rho_f) / (diam * rho_p)  # constant, [1/L]
    gprime = g * delta_rho / rho_p

    soln = (
        1
        / k
        * np.log(
            1
            + (k * vel[0] / np.sqrt(gprime * k))
            * (2 * np.arctan(vel[1] / np.sqrt(gprime / k)) - n * np.pi)
        )
    )
    return soln


def get_vel_vec(v0, theta):
    """Convenience function to return vector form
    of velocities in x and y, given some input magnitude (v0)
    and takeoff angle (theta)"""
    return np.array([v0 * np.sin(theta), v0 * np.cos(theta)])


def main():
    # set up some different initial conditions:
    vels = np.linspace(100, 250, 50)
    thetas = np.linspace(0, np.pi / 2, 50)

    # solve for L_max
    L_max = np.array(
        [[analytical(get_vel_vec(vel, theta)) for theta in thetas] for vel in vels]
    )

    # plot it
    fig, ax = plt.subplots()
    tthetas, vvels = np.meshgrid(thetas, vels)
    plt.contourf(np.rad2deg(tthetas), vvels, L_max)
    CS = plt.contour(np.rad2deg(tthetas), vvels, L_max, colors="k")
    ax.set_xlabel(r"Ejection Angle,  $\Theta$")
    ax.set_ylabel(r"Ejection velocity,  $\frac{m}{s}$")
    ax.clabel(CS, inline=1, fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
# %%
