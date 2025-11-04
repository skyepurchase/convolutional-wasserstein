import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from helpers import heat_kernel, gaussian_2d
n = 30
mean_0 = [0, 0]
mean_1 = [1, 1]
sigma = 0.1
epsilon = 0.1

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)


mu_0 = Function(V)
mu_1 = Function(V)

x, y = SpatialCoordinate(mesh)
mu_0.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean_0[0])**2 + (y - mean_0[1])**2) / (2 * sigma**2)))
mu_1.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean_1[0])**2 + (y - mean_1[1])**2) / (2 * sigma**2)))

def sinkhorn(mu_0, mu_1, tol=1e-6, maxiter=1000, epsilon=0.1):
    v_0 = Function(V)
    v_1 = Function(V)
    new_v_0 = Function(V)
    new_v_1 = Function(V)
    res = 1
    v_0.assign(1.0)

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
    while (tol < res) and (n < maxiter): 
        solv_0.solve()
        solv_1.solve()

        res = (norm(new_v_0 - v_0) + norm(new_v_1 - v_1))
        print(res)

        v_0.interpolate(new_v_0)
        v_1.interpolate(new_v_1)

        n+=1

    v_0.interpolate(epsilon * ln(v_0)) #phi
    v_1.interpolate(epsilon * ln(v_1)) #psi
    return v_0, v_1

phi, psi = sinkhorn(mu_0, mu_1)
