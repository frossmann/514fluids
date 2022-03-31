#%%
import matplotlib.pyplot as plt
from Integrators import Integrator2D, Integrator3D
import PlotUtils

# from PlotUtils import (
#     animate,
#     animate_3d,
#     plot_var_3d,
#     plot_temp_3d,
#     animate_depth_profile,
# )

"""
In progress: 
- add water for 2d case
- add water for 3d case
"""

def main_2d():
    sim = Integrator2D("/Users/francis/repos/514fluids/freeze_thaw/uservars.yaml")
    sim.build_grid()
    sim.timeloop()

    ani = PlotUtils.animate(sim)
    return sim


def main_3d():
    sim = Integrator3D("/Users/francis/repos/514fluids/freeze_thaw/uservars_3d.yaml")
    sim.build_grid()
    # plot_mask_3d(sim,0)
    sim.timeloop()
<<<<<<< HEAD
=======

    # animate_depth_profile(sim, "t")
    # animate_depth_profile(sim, "w")
>>>>>>> 49c5f9b883923e3adbdbd427ceeb761c193a2a50
    return sim


if __name__ == "__main__":
<<<<<<< HEAD
    sim = main_2d()
=======
    sim = main_3d()
    # pass

>>>>>>> 49c5f9b883923e3adbdbd427ceeb761c193a2a50

 # %%
