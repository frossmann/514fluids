#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Spatial parameters ::
# make a 2d grid
dx = 0.1  # grid spacing
L = 10  # 20m in Kessler
N = int(L // dx)  # number of grid cells per dimension
x = np.linspace(0, L, N)  # 1d axis
X, Y = np.meshgrid(x, x)  # 2d coordinates


# %% Parameters for grid variables ::
# Temperature:
T_a = -5  # degC
T_b = 0  # degC
T_g = 5  # degC
k_heat = 1e06  # m2s-1
k_surf = 5e-3  # m2yr-1

# Heave distance:
d_s = 0.6
d_v = 0.6

# Layer thicknesses:
h_stone = 0.6
h_soil = 0.4

w = 0.1  # water content
C = 0.05  # compressibility

#%% Initialize temperatures with initial conditions:
T_t0 = np.append(T_a, np.linspace(T_g, T_b, N - 1))
T = (T_t0 * np.ones_like(X)).T

# Initialize particles:
to_interface = int((h_stone * L) // dx)
P = np.ones_like(X)
P[0:to_interface] = 0
P[to_interface:] = 1


fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 10))
ax[0].imshow(T)
ax[0].set_title("Temperature grid")
ax[1].imshow(P)
ax[1].set_title("Particle grid")
ax[1].set_xlabel("Width (grid units)")
ax[0].set_ylabel("Depth (grid units)")


# %%
