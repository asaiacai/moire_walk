from os import wait
import numpy as np
import pandas as pd
from numpy.lib.type_check import real
from scipy.interpolate import interp1d
from scipy.stats import linregress

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

    def walk(self, njumps=100, nparticles=10, init=1):
        """
        Perform `njumps` of monte carlo hops with `nparticles`

        Return : np.ndarray of shape (nparticles, njumps + 1, 2)
        """
        initial_positions = np.random.randint(-init, init, size=(2, nparticles)) # Choose lattice sites
        x = initial_positions[0, :] * self.a1[0] + initial_positions[1, :] * self.a2[0]
        y = initial_positions[0, :] * self.a1[1] + initial_positions[1, :] * self.a2[1]
        initial_positions = np.array([x, y])
        tracks = np.empty((nparticles, njumps + 1, 2))
        tracks[:, 0, :] = initial_positions.T
        a_neighbors = np.random.randint(3, size=(nparticles, njumps // 2))
        b_neighbors = np.random.randint(3, size=(nparticles, njumps // 2))
        a_jumps, b_jumps = self.a_transition[a_neighbors, :], self.b_transition[b_neighbors, :]
        jumps = np.empty((nparticles, njumps, 2))
        jumps[:, ::2, :] = a_jumps
        jumps[:, 1::2, :] = b_jumps
        tracks[:, 1:, :] = jumps
        tracks = np.cumsum(tracks, axis=1)
        return tracks
     
    def get_waits(self, pot, tracks, njumps=100, nparticles=10, init=1):
        T = self.temperature
        escape_rate = np.exp(-pot.U(tracks[:, :, 0], tracks[:, :, 1]) / T)
        wait_times = np.empty((nparticles, njumps + 1))
        wait_times[:, 0] = np.zeros((nparticles))
        wait_times[:, 1:] = np.random.exponential(escape_rate)[:, :-1]
        self.mean_wait = np.mean(wait_times)
        return wait_times
        
    
    def get_tracks(self, pot, nsteps=300, njumps=100, nparticles=10, endT=1e6, init=1):
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
        tracks = self.walk(njumps, nparticles, init)
        wait_times = self.get_waits(pot, tracks, njumps=njumps, nparticles=nparticles, init=init)
        # sum of wait times gives time when molecule hops
        total_time = np.cumsum(wait_times, axis=1)
        
        # get evenly spaced timesteps as if in lab setting
        if endT:
            tOut = np.linspace(0, endT, nsteps + 1) # time value
        else:
            endT = self.mean_wait * 100
            tOut = np.linspace(0, endT, nsteps + 1) # time value
        real_space_tracks = np.empty(shape=(nparticles, nsteps + 1, 2))
        for particle in range(nparticles):
            for i in [0, 1]:
                real_space_tracks[particle, :, i] = interp1d(total_time[particle, :], tracks[particle, :, i], 
                                                    kind='previous', 
                                                    bounds_error=False)(tOut)

        df = pd.DataFrame()
        r_squared = (real_space_tracks[:, :, 0] - real_space_tracks[:,np.newaxis, 0, 0]) ** 2 + (real_space_tracks[:, :, 1] - real_space_tracks[:, np.newaxis, 0, 1]) ** 2
        for particle in range(nparticles):
            tmp = pd.DataFrame(real_space_tracks[particle], columns=['x', 'y'])
            tmp['particle'] = particle
            tmp['time'] = tOut
            tmp['r^2'] = r_squared[particle, :]
            df = pd.concat([df, tmp])
        return df
