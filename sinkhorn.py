from firedrake import *

from solvers import HeatEquationSolver


# Constants
N = 30
MEAN_0 = [0.1, 0.1]
MEAN_1 = [0.5, 0.5]
SIGMA = 0.1
EPSILON = 0.1
MESH = UnitSquareMesh(N, N)
V = FunctionSpace(MESH, "CG", 1)

# Set up distributions
mu_0 = Function(V)
mu_1 = Function(V)

x, y = SpatialCoordinate(MESH)
mu_0.interpolate((1 / (2 * pi * SIGMA**2)) * exp(-((x- MEAN_0[0])**2 + (y - MEAN_0[1])**2) / (2 * SIGMA**2)))
mu_1.interpolate((1 / (2 * pi * SIGMA**2)) * exp(-((x- MEAN_1[0])**2 + (y - MEAN_1[1])**2) / (2 * SIGMA**2)))
# Normalise
Imu_0 = assemble(mu_0*dx)
Imu_1 = assemble(mu_1*dx)
mu_1.assign(mu_1/Imu_1)
mu_0.assign(mu_0/Imu_0)


def sinkhorn(mu_0, mu_1, tol=1e-6, maxiter=10, epsilon=0.1):
    phi = Function(V)
    psi = Function(V)

    Solver_0 = HeatEquationSolver(V)
    Solver_1 = HeatEquationSolver(V)
    Solver_1.initialise()

    n=0
    res = 1
    while (tol < res) and (n < maxiter):
        Solver_1.solve()
        res = norm(
            Solver_0.function -
            (mu_0 / Solver_1.output_function)
        )
        Solver_0.update(mu_0 / Solver_1.output_function)
        Solver_0.solve()
        Solver_1.update(mu_1 / Solver_0.output_function)

        print(res)
        n+=1

    # is this -1 * epsilon?
    phi.interpolate(epsilon * ln(Solver_0.function)) #phi
    psi.interpolate(epsilon * ln(Solver_1.function)) #psi
    return phi, psi

phi, psi = sinkhorn(mu_0, mu_1, epsilon=EPSILON)

Vc = MESH.coordinates.function_space()
x, y = SpatialCoordinate(MESH)
f = Function(Vc).interpolate(grad(phi))
MESH.coordinates.assign(f)

VTKFile("phi.pvd").write(mu_0)
