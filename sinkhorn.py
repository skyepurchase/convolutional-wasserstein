import matplotlib.pyplot as plt

from firedrake import *
from firedrake.pyplot import tripcolor, tricontour

from solvers import HeatEquationSolver


def initialise_env(epsilon, mesher, make_space):
    """Initialise mesh and function space. Mesh discretisation fixed to be of order 1/(epsilon**2)"""
    n = int(1 / (epsilon**2)) # Mesh size
    mesh = mesher(30, 30)
    v = make_space(mesh, "CG", 1)
    return mesh, v

# Constants
EPSILON = 0.01
MEAN_0 = [0.1, 0.1]
MEAN_1 = [0.5, 0.5]
SIGMA = 0.1

# Initialise mesh and function space
MESH, V = initialise_env(EPSILON, UnitSquareMesh, FunctionSpace)

# Set up probability distributions
mu_0 = Function(V)
mu_1 = Function(V)

x, y = SpatialCoordinate(MESH)
mu_0.interpolate((1 / (2 * pi * SIGMA**2)) * exp(-((x- MEAN_0[0])**2 + (y - MEAN_0[1])**2) / (2 * SIGMA**2)))
mu_1.interpolate((1 / (2 * pi * SIGMA**2)) * exp(-((x- MEAN_1[0])**2 + (y - MEAN_1[1])**2) / (2 * SIGMA**2)))

# Normalise on mesh
Imu_0 = assemble(mu_0*dx)
Imu_1 = assemble(mu_1*dx)
mu_1.assign(mu_1/Imu_1)
mu_0.assign(mu_0/Imu_0)


def sinkhorn(
    mu_0,
    mu_1,
    V,
    tol=1e-6,
    maxiter=10,
    epsilon=0.1
):
    """
    Repeat the sinkhorn iteration until the tolerance is reached or maximum iterations.

    Parameters
    ----------
    mu_0    : The source distribution
    mu_1    : The target distribution
    V       : The function space mu_0 and mu_1 are in
    tol     : The tolerance at which to stop
    maxiter : The maximum number of iterations
    epsilon : The regularisation parameter
    """
    phi = Function(V)
    psi = Function(V)

    Solver_0 = HeatEquationSolver(V, dt=epsilon/2)
    Solver_1 = HeatEquationSolver(V, dt=epsilon/2)
    Solver_1.initialise()

    i = 0
    res = 1
    while (tol < res) and (i < maxiter):
        Solver_1.solve()
        res = norm(
            Solver_0.function -
            (mu_0 / Solver_1.output_function)
        )
        Solver_0.update(mu_0 / Solver_1.output_function)
        Solver_0.solve()
        Solver_1.update(mu_1 / Solver_0.output_function)

        print(res)
        i += 1

    phi.interpolate(epsilon * ln(Solver_0.function)) #phi
    psi.interpolate(epsilon * ln(Solver_1.function)) #psi
    return phi, psi

phi, psi = sinkhorn(mu_0, mu_1, V, epsilon=EPSILON)

Vc = MESH.coordinates.function_space()
x, y = SpatialCoordinate(MESH)
f = Function(Vc).interpolate(grad(phi))
MESH.coordinates.assign(f)

fig, axes = plt.subplots()
colors = tripcolor(mu_0, axes=axes)
fig.colorbar(colors)
plt.show()
