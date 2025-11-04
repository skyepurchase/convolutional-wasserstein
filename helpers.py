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