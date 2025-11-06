from firedrake import *
from firedrake.pyplot import tripcolor

import matplotlib.pyplot as plt
from scipy.special import iv # Bessel function 
import numpy as np
from sinkhorn import sinkhorn

#Constants 
EPSILON = 0.1
N = 100
R = 1
KAPPA_0 = KAPPA_1 = 20
MEAN_0 = 0
MEAN_1 =1

# Circle mesh
MESH = CircleManifoldMesh(N, radius=R)
V = FunctionSpace(MESH, "CG", 1)

# Set up probability distributions
mu_0 = Function(V)
mu_1 = Function(V)

x, y = SpatialCoordinate(MESH)
theta = atan2(y, x)

#Von Mises density
mu_0.interpolate(exp(KAPPA_0 * cos(theta - MEAN_0)) / (2.0 * pi * iv(0, KAPPA_0)))
mu_1.interpolate(exp(KAPPA_1 * cos(theta - MEAN_1)) / (2.0 * pi * iv(0, KAPPA_1)))

# Normalise on mesh
Imu_0 = assemble(mu_0*dx)
Imu_1 = assemble(mu_1*dx)
mu_1.assign(mu_1/Imu_1)
mu_0.assign(mu_0/Imu_0)


# Run Sinkhorn
phi, psi = sinkhorn(mu_0, mu_1, V, epsilons=[EPSILON], maxiter=100)

# Plot transport map 
Vc = MESH.coordinates.function_space()
grad_phi_fun = Function(Vc).interpolate(grad(phi))
VTKFile("grad_phi.pvd").write(grad_phi_fun)
