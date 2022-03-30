#%%
import matplotlib.pyplot as plt
from Integrators import Integrator2D, Integrator3D
import PlotUtils as plot
from matplotlib.animation import FFMpegWriter

# import animate, animate_3d, plot_var_3d, plot_temp_3d, animate_depth_profile


def main_2d():
    sim = Integrator2D("/Users/francis/repos/514fluids/freeze_thaw/uservars.yaml")
    sim.build_grid()
    sim.timeloop()

    ani = plot.animate(sim)
    return sim


def main_3d():
    sim = Integrator3D("/Users/francis/repos/514fluids/freeze_thaw/uservars_3d.yaml")
    sim.build_grid()
    sim.timeloop()

    # ani = animate_3d(sim)
    ani = plot.animate_depth_profile(sim, "w")
    ani = plot.animate_depth_profile(sim, "t")
    # ani.save("movie.mp4")

    # writer = FFMpegWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    return sim


if __name__ == "__main__":
    sim = main_3d()


# %%

# %%
