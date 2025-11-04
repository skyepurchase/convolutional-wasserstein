from firedrake import *

def heat_kernel(T, N, mesh):
    """
    Solves the heat equation on given mesh given time horizon T and number of timesteps N.
    Uses implicit Euler time-stepping scheme.
    Returns the solution at time T.

    Parameters
    ----------
    T : float
        Total time horizon.
    N : int
        Number of timesteps.
    mesh : Mesh
    """
    dt = Constant(T / N)
    V = FunctionSpace(mesh, "CG", 1)
    un = Function(V) #Initial condition (sol. at previous timestep)
    un.assign(1.0) 

    u = TrialFunction(V)
    v = TestFunction(V)

    a = (dt * inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(un, v) * dx

    unp1 = Function(V) #Redefine to be a function holding the solution:

    params = {'ksp_type':'preonly', 'pc_type':'lu'}
    prob = LinearVariationalProblem(a, L, unp1)
    solv = LinearVariationalSolver(prob, solver_parameters=params)
    solv.solve()
    
    return unp1 


def gaussian_2d(n, mu, sigma):
    """
    Creates a 2D Gaussian distribution on an n x n grid with mean mu and standard deviation sigma.

    Parameters
    ----------
    n : int
        Size of the grid (n x n).
    mu : tuple of float
        Mean of the Gaussian in x and y directions.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    numpy.ndarray
        2D array representing the Gaussian distribution.
    """
    import numpy as np

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    gauss = (1 / (2 * np.pi * sigma**2)) * np.exp(-((X - mu[0])**2 + (Y - mu[1])**2) / (2 * sigma**2))
    
    return gauss