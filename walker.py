import numpy as np

# rotation matrix
r = lambda theta : np.array(( (np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta)) ))

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
        AB = np.array([1/np.sqrt(3), 0]) * lattice_constant
        self.a_transition = np.array([
            AB,
            r(deg(120)) @ AB,
            r(deg(240)) @ AB
        ], dtype=np.float64)
        self.b_transition = -a_transition

    def walk(self, njumps=100, nparticles=10):
        """
        Perform `njumps` of monte carlo hops with `nparticles`
        """
        initial_positions = np.random.randint(-10, 10, size=(2, nparticles)) # Choose lattice sites
        x = initial_positions[0, :] * a1[0] + initial_positions[1, :] * a2[0]
        y = initial_positions[0, :] * a1[1] + initial_positions[1, :] * a2[1]
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
        T = self.temperature
        tracks = self.walk(njumps, nparticles)
        escape_rate = np.exp(U.pot(tracks) / T)
        