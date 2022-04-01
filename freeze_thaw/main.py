#%%
import matplotlib
import matplotlib.pyplot as plt
from Integrators import Integrator2D, Integrator3D
import PlotUtils
import numpy as np

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

    # animate_depth_profile(sim, "t")
    # animate_depth_profile(sim, "w")
    return sim


if __name__ == "__main__":
    sim = main_2d()

# %%

n1 = 1731
n2 = n1 + 1
fig, ax = plt.subplots(3)
im = ax[0].imshow(sim.T[:, :, n1])
plt.colorbar(ax=ax[0], mappable=im)

ax[1].hist(sim.T[:, :, n1].ravel())
ax[2].plot(sim.T[:, 0, n1])
ax[0].set_title("n1")
ax[1].set_title("distribution")
ax[2].set_title("temp depth profile")
plt.tight_layout()

fig, ax = plt.subplots(3)
im = ax[0].imshow(sim.T[:, :, n2])
plt.colorbar(ax=ax[0], mappable=im)

ax[1].hist(sim.T[:, :, n2].ravel())
ax[2].plot(sim.T[:, 0, n2])
ax[0].set_title("n2")
ax[1].set_title("distribution")
ax[2].set_title("temp depth profile")
plt.tight_layout()

idx = sim.find_freezing(sim.W[:, :, n1], sim.T[:, :, n2])
# %%
times = [int(time) for time in np.linspace(0, sim.end, 10)]
plt.figure(figsize=(10, 10))
for time in times:
    plt.plot(sim.T[:, 0, time], label=time)
plt.hlines(0, 0, 22, linestyles="--", color="k")
plt.vlines(0.6 * sim.ny, -5, 5, linestyles=":", color="r")
plt.legend()
# %%
