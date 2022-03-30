#%%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def animate(sim, clim=[-5, 5]):

    fig = plt.figure()  # make figure
    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(sim.T[:, :, 0], vmin=clim[0], vmax=clim[1])
    ax = fig.gca()
    ax.set_xlabel("Length")
    ax.set_ylabel("Depth")
    # ax.axis('equal')
    # theTitle = "fig. {}: 2D Heat Diffusion from a Line Source, t={}".format(
    #     ij + 1, sim.dt * time_idx[ij]
    # )
    # ax.set_title(theTitle)
    cbar = plt.colorbar(im)
    cbar.set_label("Temperature", rotation=270)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(sim.T[:, :, j])
        theTitle = f"t={((sim.dt * j) / (60 * 60 * 24)):.2f} days"
        ax.set_title(theTitle)
        # return the artists set
        return [im]

    # kick off the animation
    ani = animation.FuncAnimation(
        fig, updatefig, frames=np.arange(0, sim.timevars.nt, 100), interval=1, blit=True
    )
    plt.show()
    return [ani]


def plot_var_3d(sim, var, idx):
    if var.lower() == "t":
        plotvar = sim.T
    if var.lower() == "m":
        plotvar = sim.M
    if var.lower() == "w":
        plotvar = sim.W

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth")
    ax.invert_zaxis()
    sc = ax.scatter(
        sim.X,
        sim.Y,
        sim.Z,
        c=plotvar[:, :, :, idx],
        alpha=0.5,
    )
    cbar = plt.colorbar(sc)
    plt.show()


def plot_temp_3d(sim, idx, clim=[-5, 5]):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth")
    ax.invert_zaxis()
    sc = ax.scatter(
        sim.X,
        sim.Y,
        sim.Z,
        c=sim.T[:, :, :, idx],
        alpha=0.5,
        vmin=clim[0],
        vmax=clim[1],
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature")
    plt.show()


def animate_3d(sim, clim=[-5, 5], step=5):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth")
    ax.invert_zaxis()
    sc = ax.scatter(
        sim.X[::step, ::step, ::step],
        sim.Y[::step, ::step, ::step],
        sim.Z[::step, ::step, ::step],
        c=sim.T[::step, ::step, ::step, 0].ravel(),
        alpha=0.5,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature", rotation=270)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        sc.set_array(sim.T[::step, ::step, ::step, j].ravel())
        theTitle = f"t={((sim.dt * j) / (60 * 60 * 24)):.2f} days"
        ax.set_title(theTitle)
        # return the artists set
        return [sc]

    # # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=np.arange(0, sim.nt, 100),
        interval=1,
        blit=True,
    )
    plt.show()
    return [ani]
    # return None


def animate_depth_profile(sim, var, xi=0, yi=0):
    """
    Hello
    """
    if var.lower() == "t":
        plotvar = sim.T
    if var.lower() == "m":
        plotvar = sim.M
    if var.lower() == "w":
        plotvar = sim.W

    fig, ax = plt.subplots()
    (line,) = ax.plot(sim.dz * np.arange(0, sim.nz), plotvar[xi, yi, :, 0])

    ax.set_xlabel("Depth, (m)")
    ax.set_ylabel("Temperature, (degC)")

    def updateFig(j):
        line.set_ydata(plotvar[xi, yi, :, j])
        return [line]

    ani = animation.FuncAnimation(
        fig, updateFig, frames=np.arange(0, sim.end, 10), interval=5, blit=True
    )
    plt.show()
    return ani


# %%
