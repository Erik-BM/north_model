# This model is described in North et. al., 1981.
# The addition of a deep ocean coupling is done by Erik B. Myklebust (2017),
# see appendix A in thesis for details.
#
#
#
#
# This program requires some packages. Listed below
# numpy
# scipy
# tqdm (progress indicator)
#

import numpy as np
import scipy as sp
from scipy.integrate import ode
from scipy import linalg
from tqdm import *
from scipy.interpolate import interp1d


class north_model:

    def run_model(self, time):

        def spaital_dep(t, x):
            # Incoming solar radiation
            S0 = 420  # Radiation at eq.
            S1 =  338  # Seasonality
            S2 = 240  # Spatial dep.
            tmp = S0 - S1 * x * np.cos(2 * np.pi * t) - S2 * x**2
            return tmp

        def coalbedo(x, x_e):
            # Returning the smooth co-albedo function
            b_0 = 0.4  # co-albedo over ice
            a_0 = 0.7  # co-albedo over water
            a_2 = -0.1  # spatial dep.
            gamma = 150  # Steepness
            out = (a_0 + a_2 * x**2 - b_0) / \
                (np.exp(gamma * (x - x_e)) + 1) + b_0
            return out

        def initial_condition(x):
            return 7.5 + 20 * (1 - 2 * x**2)

        def construct_matrix():
            # See Bitz and Row (2009)
            xb = np.arange(0, 1 - self.dx, self.dx)
            lam = 1 / self.dx**2 * (1 - xb**2)
            L1 = np.append(0, -lam)
            L2 = np.append(-lam, 0)
            L3 = -L1 - L2
            diffop = - \
                np.diag(L3) - np.diag(L2[:self.size - 1],
                                      1) - np.diag(L1[1:self.size], -1)
            diffop = self.D / self.cw * diffop
            deep_op = -1 * np.diag(self.kappa * np.ones(self.size))
            deep_op = deep_op / self.cd
            # Block matrix consisting of diffusion and deep ocean eq.
            return np.bmat([[diffop, np.zeros([self.size, self.size])], [np.zeros([self.size, self.size]), deep_op]])

        def find_nearest(array, value):
            # Finding the index in array nearest to value
            idx = (np.abs(array - value)).argmin()
            return idx

        def ice_edge(temp):
            # Ice edge defined as grid point closest to 0 degrees
            idx = find_nearest(temp, 0)
            out = self.xgrid[idx]
            self.ice = out
            return self.alb[idx]

        def split_array(res):
            # Separate a matrix into two pieces
            out1 = []
            out2 = []
            for v in res:
                out1.append(v[:int(len(v) / 2)])
                out2.append(v[int(len(v) / 2):])
            return np.asarray(out1), np.asarray(out2)

        def to_solver(time, vector):
            # See appendix A in thesis
            f = self.forcing(time)

            half = int(len(vector) / 2)

            albedo = ice_edge(vector[:half])
            nonlin_part = (spaital_dep(time, np.append(
                self.xgrid, self.xgrid)) * np.append(albedo, albedo)) * self.temp_vec / self.cw
            inhom_part = (f - self.A - (self.B + np.append(self.kappa,
                                                           self.kappa)) * vector) / self.cw * self.temp_vec

            kappa_vec = np.append(
                vector[half:] / self.cw, vector[:half] / self.cd)

            tmp = self.matrix_op(vector.T) + nonlin_part + inhom_part + \
                np.append(self.kappa, self.kappa) * kappa_vec
            return tmp

        # Difference matrix set up
        self.matrix = sp.sparse.dia_matrix(construct_matrix())
        self.matrix_op = self.matrix.dot
        self.alb = list(map(lambda xe: coalbedo(self.xgrid, xe), self.xgrid))
        self.temp_vec = np.ones(len(self.xgrid) * 2)
        self.temp_vec[int(len(self.temp_vec) / 2):] = 0
        self.deep_temp_vec = np.ones(len(self.xgrid) * 2)
        self.deep_temp_vec[:int(len(self.deep_temp_vec) / 2)] = 0

        # Checking for spin up
        if hasattr(self, 'temp'):
            tmp = self.temp[-1, :]
        elif hasattr(self, 'init'):
            tmp = self.init
        else:
            tmp = initial_condition(self.xgrid)

        # Setting up solver
        backend = 'lsoda'
        solver = ode(to_solver).set_integrator(backend)
        initial = tmp
        initial = np.append(initial, initial)
        solver.set_initial_value(initial, 0)

        # Defining output arrays
        result = [initial]
        self.ice = 0

        result_append = result.append
        self.ice_edge = []
        ice_append = self.ice_edge.append

        # Progress bar
        pbar = tqdm(total=time)
        i = 1

        # Main loop
        while solver.successful() and solver.t < time:
            next_solution = solver.integrate(solver.t + self.dt)
            if np.floor(solver.t) >= i:
                #saving one value per year
                i += 1
                pbar.update(1)
                result_append(next_solution)
                ice_append(self.ice)
        pbar.close()

        # Separating temp array from deep temp array
        self.temp, self.deep_temp = split_array(result)

        self.ice_edge = np.asarray(self.ice_edge)

    def __init__(self, grid_size=400, f=np.zeros(100), load_from_init=True):
        # grids
        self.dx = 1.0 / grid_size
        self.time_steps = 100
        self.dt = 1.0 / self.time_steps
        self.tmax = len(f)
        self.size = grid_size

        self.ma_length = ma_length

        self.xgrid = np.arange(self.dx / 2, 1 + self.dx / 2, self.dx)

        # constants
        self.D = 0.66
        self.B = 2.4
        self.A = 188.5  # historic forcing
        self.cw = 7.3

        # Deep ocean
        self.cd = 106
        self.taud = 30
        self.kappa = 5 * 0.73 * (-np.tanh(10 * (self.xgrid - 0.2)) + 1) / 2

        # Checking for initial conditions in file
        import os.path
        if load_from_init == True and os.path.isfile("initial.csv") == True:
            print("Loading initial from file")
            self.init = np.loadtxt("initial.csv", delimiter=',')
        else:
            print("Spinning up")
            spinup = 2000
            forc = f[0] * np.ones(2 * spinup)
            # interpolating forcing
            self.forcing = interp1d(range(0, len(forc)), forc)
            self.run_model(spinup)
            np.savetxt("initial.csv", np.asarray(
                self.temp[-1, :]), delimiter=',')
            print(np.mean(self.temp[-1, :]))
            print(self.ice_edge[-1])

        forc = np.append(f, f[-1] * np.ones(len(f)))

        # interpolating forcing
        self.forcing = interp1d(range(0, len(forc)), forc)

        print("Running")
        self.run_model(self.tmax)
