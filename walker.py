from os import wait
import numpy as np
import pandas as pd
from numpy.lib.type_check import real
from scipy.stats import linregress
import pdb
from tqdm import tqdm

# rotation matrix
rotate = lambda theta : np.array(( (np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta)) ))
deg2rads = lambda deg : deg * np.pi / 180

class Walker:
    """
    defines the experimental parameters for the random walker
    """

    def __init__(self, lattice_constant, temperature):
        self.lattice_constant = lattice_constant # in terms of nm
        self.temperature = temperature # in terms of Kelvin

class Graphene_Walker(Walker):

    def __init__(self, lattice_constant, temperature):
        super().__init__(lattice_constant, temperature)
        # primitive lattice vectors
        self.a1 = np.array([np.sqrt(3)/2, 1/2]) * self.lattice_constant
        self.a2 = np.array([np.sqrt(3)/2, -1/2]) * self.lattice_constant
        # nearest neighbor vectors
        AB = np.array([1/np.sqrt(3), 0]) * self.lattice_constant
        self.a_transition = np.array([
            AB,
            rotate(deg2rads(120)) @ AB,
            rotate(deg2rads(240)) @ AB
        ], dtype=np.float64)
        self.b_transition = -self.a_transition

    def walk(self, njumps=100, nparticles=10, init=0):
        """
        Perform `njumps` of monte carlo hops with `nparticles`

        Return : np.ndarray of shape (nparticles, njumps + 1, 2)
        """
        if init != 0:
            initial_positions = np.random.randint(-init, init, size=(2, nparticles)) # Choose lattice sites
        else:
            initial_positions = np.zeros((2, nparticles)) # Choose lattice sites
        x = initial_positions[0, :] * self.a1[0] + initial_positions[1, :] * self.a2[0]
        y = initial_positions[0, :] * self.a1[1] + initial_positions[1, :] * self.a2[1]
        initial_positions = np.array([x, y])
        a_jumps = self.a_transition[np.random.randint(3, size=(nparticles, njumps // 2)), :]
        b_jumps = self.b_transition[np.random.randint(3, size=(nparticles, njumps // 2)), :]
        jumps = np.empty((nparticles, njumps+1, 2))
        jumps[:, 0, :] = initial_positions.T
        jumps[:, 1::2, :] = a_jumps
        jumps[:, 2::2, :] = b_jumps
        return np.cumsum(jumps, axis=1)
     
    def get_waits(self, pot, tracks, njumps=100, nparticles=10, init=0):
        T = self.temperature
        escape_rate = np.exp(-pot.U(tracks[:, :, 0], tracks[:, :, 1]) / T)
        wait_times = np.empty((nparticles, njumps + 1))
        wait_times[:, 0] = np.zeros((nparticles))
        wait_times[:, 1:] = np.random.exponential(escape_rate)[:, :-1]
        self.mean_wait = np.mean(wait_times)
        return wait_times
        
    
    def get_tracks(self, pot, nsteps=300, njumps=100, nparticles=10, endT=1e6, init=0):
        """
        get `nsteps` timesteps of random walk with `njumps` monte carlo sim
        for `nparticles`

        U : Lattice Potential instance
        nsteps : number of evenly spaced timestamps to sample trajectories up `endT`
        njumps : number of monte carlo hops
        nparticles : number of particles
        endT : the time range to sample trajectories

        Return : np.ndarray array of shape (nparticles, nsteps + 1, 2)
        """
        nsteps += 1
        # get evenly spaced timesteps as if in lab setting
        if endT:
            tFinal = np.linspace(0, endT, nsteps + 1) # time value
            idx = np.arange(nsteps + 1)
#         else:
#             endT = self.mean_wait * njumps
#             tOut = np.linspace(0, endT, nsteps + 1) # time value
        
        real_space_tracks = np.empty((nparticles, nsteps + 1, 2))
        last_time = np.zeros((nparticles, 1))
        last_pos = np.zeros((nparticles, 1, 2))
        tracks = np.zeros((nparticles, njumps+1, 2))
        wait_times = np.zeros((nparticles, njumps+1))
        
        curr_end = 0
        while curr_end < endT:
            tracks[:] = self.walk(njumps, nparticles, init) + last_pos
            init = 0
            wait_times[:] = self.get_waits(pot, tracks, njumps=njumps, nparticles=nparticles)
            wait_times[:] = np.cumsum(wait_times, axis=1) + last_time
            curr_end = np.min(wait_times[:, -1])
            included = tFinal < curr_end
            tFinal, idx, tOut, iOut = tFinal[~included], idx[~included], tFinal[included], idx[included]
            for particle in range(nparticles):
                last_idx = np.argwhere(wait_times[particle] < curr_end)[-1]
                last_time[particle] = wait_times[particle, last_idx]
                last_pos[particle, :] = tracks[particle, last_idx, :]
                j = np.searchsorted(wait_times[particle, :], tOut, side='right') - 1
                real_space_tracks[particle, iOut, :] = tracks[particle, j, :]
        dfs = []
        tFinal = np.linspace(0, endT, nsteps + 1)
        for particle in range(nparticles):
            tmp = pd.DataFrame(real_space_tracks[particle, 1:], columns=['x', 'y'])
            tmp['particle'] = particle
            tmp['time'] = tFinal[1:]
            dfs.append(tmp)
        print("{} done".format(self.temperature))
        return pd.concat(dfs)