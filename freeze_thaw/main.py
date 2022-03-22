#%%
import matplotlib.pyplot as plt
from Integrators import Integrator2D, Integrator3D
from PlotUtils import animate, animate_3d, plot_mask_3d, plot_temp_3d


def main_2d():
    sim = Integrator2D("/Users/francis/repos/514fluids/freeze_thaw/uservars.yaml")
    sim.build_grid()
    sim.timeloop()

    ani = animate(sim)
    return sim


def main_3d():
    sim = Integrator3D("/Users/francis/repos/514fluids/freeze_thaw/uservars_3d.yaml")
    sim.build_grid()
    # plot_mask_3d(sim,0)
    sim.timeloop()
    return sim


if __name__ == "__main__":
    sim = main_3d()
