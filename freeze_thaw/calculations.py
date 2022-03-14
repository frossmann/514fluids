#%%
import numpy as np
import matplotlib.pyplot as plt


class GridCell():
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
                self.domain = 'soil'
            else:
                self.domain = 'stone'


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
        if self.domain == 'stone': return

        if self.last_temp + delta_T <= 0: 
            if self.w == 0:
                self.frozen = True 
            else: 
                joules = self.get_latent_heat(delta_T)
        elif self.tas

    







# #%% Spatial parameters ::
# # make a 2d grid
# dx = 0.1  # grid spacing
# L = 10  # 20m in Kessler
# N = int(L // dx)  # number of grid cells per dimension
# x = np.linspace(0, L, N)  # 1d axis
# X, Y = np.meshgrid(x, x)  # 2d coordinates




#%%# %%
