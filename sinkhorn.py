import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from helpers import heat_kernel, gaussian_2d
n = 30
mean_0 = [0.1, 0.1]
mean_1 = [0.5, 0.5]
sigma = 0.1
epsilon = 0.01

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)


mu_0 = Function(V)
mu_1 = Function(V)

x, y = SpatialCoordinate(mesh)
mu_0.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean_0[0])**2 + (y - mean_0[1])**2) / (2 * sigma**2)))
mu_1.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean_1[0])**2 + (y - mean_1[1])**2) / (2 * sigma**2)))
Imu_0 = assemble(mu_0*dx)
Imu_1 = assemble(mu_1*dx)
mu_1.assign(mu_1/Imu_1)
mu_0.assign(mu_0/Imu_0)

def sinkhorn(mu_0, mu_1, tol=1e-6, maxiter=1000, epsilon=0.1):
    v_0 = Function(V)
    v_1 = Function(V)
    new_v_0 = Function(V)
    new_v_1 = Function(V)
    res = 1
    v_0.assign(1.0)
    v_1.assign(1.0) # so we don't get -inf errors

    u = TrialFunction(V)
    v = TestFunction(V)

    dt = epsilon
    a = (dt * inner(grad(u), grad(v)) + inner(u, v)) * dx
    L_0 = inner(v_0, v) * dx
    L_1 = inner(v_1, v) * dx

    params = {'ksp_type':'preonly', 'pc_type':'lu'}
    prob_0 = LinearVariationalProblem(a, L_0, new_v_0)
    solv_0 = LinearVariationalSolver(prob_0, solver_parameters=params)
    prob_1 = LinearVariationalProblem(a, L_1, new_v_1)
    solv_1 = LinearVariationalSolver(prob_1, solver_parameters=params)

    n=0
    res = 1
    maxiter = 10
    while (tol < res) and (n < maxiter):

        old_0 = v_0.copy(deepcopy=True)
        old_1 = v_1.copy(deepcopy=True)

        solv_1.solve()
        v_0.interpolate(mu_0 / new_v_1)
        solv_0.solve()
        v_1.interpolate(mu_1 / new_v_0)
        
        res = norm(v_0 - old_0) + norm(v_1 - old_1)
        print(res)
        n+=1

    # is this -1 * epsilon?
    v_0.interpolate(epsilon * ln(v_0)) #phi
    v_1.interpolate(epsilon * ln(v_1)) #psi
    return v_0, v_1

phi, psi = sinkhorn(mu_0, mu_1)

Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(grad(phi))
mesh.coordinates.assign(f)

# plan = Function(V)
# plan.interpolate(phi * psi) # ignore ; this is a test
# x, y = SpatialCoordinate(mesh)

# fig, axes = plt.subplots()
# colors = tripcolor(plan, axes=axes)
# fig.colorbar(colors)
# plt.title("Sinkhorn Plan")
# plt.show()

VTKFile("phi.pvd").write(mu_0)

