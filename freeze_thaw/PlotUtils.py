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


def plot_mask_3d(sim, idx, clim=[0, 2]):
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
        c=sim.M[:, :, :, idx],
        alpha=0.5,
        vmin=clim[0],
        vmax=clim[1],
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


def animate_3d(sim, clim=[-5, 5]):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth")
    ax.invert_zaxis()
    sc = ax.scatter(sim.X, sim.Y, sim.Z, c=sim.T[:, :, :, 0].ravel(), alpha=0.5)
    cbar = plt.colorbar(sc)
    cbar.set_label("Temperature", rotation=270)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        sc.set_array(sim.T[:, :, :, j].ravel())
        theTitle = f"t={((sim.dt * j) / (60 * 60 * 24)):.2f} days"
        ax.set_title(theTitle)
        # return the artists set
        return [sc]

    # # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=np.arange(0, sim.timevars.nt, 100),
        interval=1,
        blit=False,
    )
    plt.show()
    return [ani]
    # return None
