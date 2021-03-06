import numpy as np

class Triangular_Lattice:
    
    def __init__(self, lattice_constant, amplitude):
        self.lattice_constant = lattice_constant
        self.amplitude = amplitude # in terms of meV
        # convert to kelvin
        self.amplitude *= 11.605

    def U(self, x, y):
        """
        evaluates potential at x, y coord lists
        x : list of x pos for [nparticles, nsteps]
        y : list of y pos for [nparticles, nsteps]
        shift in energy is for numerical stability, since constant shift in energy
        only multiplies all rates by a common scale factor
        """
        a, A = self.lattice_constant, self.amplitude
        theta_x, theta_y = 2 * np.pi * x / a, 2 * np.pi * y / (a * np.sqrt(3))
        sigma = 2 * np.cos(theta_x) * np.cos(theta_y) + np.cos(2 * theta_y)
        return A * 2 / 9 * ((3 - sigma))
        
        
        
