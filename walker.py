import numpy as np
from numpy.lib.type_check import real
from scipy.interpolate import interp1d

# rotation matrix
rotate = lambda theta : np.array(( (np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta)) ))
deg2rads = lambda deg : np.pi / 180

class Walker:
    """
    defines the experimental parameters for the random walker
    """

    def __init__(self, lattice_constant, temperature):
        self.lattice_constant = lattice_constant
        self.temperature = temperature

class Graphene_Walker(Walker):

    def __init__(self, lattice_constant, temperature):
        super.__init__(lattice_constant, temperature)
        # primitive lattice vectors
        self.a1 = np.array([np.sqrt(3)/2, 1/2]) * lattice_constant
        self.a2 = np.array([np.sqrt(3)/2, -1/2]) * lattice_constant
        # nearest neighbor vectors
        AB = np.array([1/np.sqrt(3), 0]) * lattice_constant
        self.a_transition = np.array([
            AB,
            rotate(deg2rads(120)) @ AB,
            rotate(deg2rads(240)) @ AB
        ], dtype=np.float64)
        self.b_transition = -self.a_transition

    def walk(self, njumps=100, nparticles=10):
        """
        Perform `njumps` of monte carlo hops with `nparticles`

        Return : np.ndarray of shape (nparticles, njumps + 1, 2)
        """
        initial_positions = np.random.randint(-100, 100, size=(2, nparticles)) # Choose lattice sites
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
    
    def get_tracks(self, U, nsteps=300, njumps=100, nparticles=10):
        """
        get `nsteps` timesteps of random walk with `njumps` monte carlo sim
        for `nparticles`

        U : Lattice Potential instance
        nsteps : number of timesteps
        njumps : number of monte carlo hops
        nparticles : number of particles

        Return : np.ndarray array of shape (nparticles, nsteps + 1, 2)
        """
        T = self.temperature
        tracks = self.walk(njumps, nparticles)
        escape_rate = np.exp(U.pot(tracks) / T)
        wait_times = np.empty((nparticles, njumps + 1))
        wait_times[:, 0] = np.zeros((nparticles))
        wait_times[:, 1:] = np.random.exponential(escape_rate)[:, :-1]
        
        # sum of wait times to gives time when molecule hops
        total_time = np.cumsum(wait_times, axis=1)
        # rescale trajectories in time (OK because we're just estimating power law exponents which are scale invariant)
        # we rescale anyway for deep learning
        total_time_normed = total_time / np.linalg.norm(total_time, np.inf, axis=1)[:, np.newaxis] * nsteps
        
        # get evenly spaced timesteps as if in lab setting
        tOut = np.arange(0, nsteps + 1, 1, dtype=np.uint8) # frame number
        real_space_tracks = np.empty(shape=(nparticles, nsteps + 1, 2))
        for particle in range(nparticles):
            for i in [0, 1]:
                real_space_tracks[particle, :, i] = interp1d(total_time_normed[particle, :], tracks[particle, :, i], 
                                                    kind='next', 
                                                    bounds_error=False)(tOut)

        return real_space_tracks