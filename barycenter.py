from firedrake import *
from solvers import HeatEquationSolver
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor

def wasserstein_barycenter(mus, alphas):
    """
    Compute the Wasserstein barycenter of given distributions.
    """

    num_dists = len(mus) # number of input distributions
    try:
        assert abs(sum(alphas)) == 1, "Weights must sum to 1."
    except AssertionError as e:
        print("Error in weights: ", e)
        raise e
    
    epsilon = 1
    V = FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)

    mu = Function(V).assign(1.0)

    v_list = [HeatEquationSolver(V, dt=epsilon/2) for _ in range(num_dists)]
    w_list = [HeatEquationSolver(V, dt=epsilon/2) for _ in range(num_dists)]
    d_list = [Function(V).assign(1.0) for _ in range(num_dists)]

    for i in range(num_dists):
        v_list[i].initialise()
        w_list[i].initialise()
    
    # Placeholder for barycenter computation logic

    curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]
    for j in range(num_dists):
        
        # THIS LOOP CAN BE PARALLELISED
        for i in range(num_dists):
            v_list[i].solve()
            w_list[i].update(curr[i] / v_list[i].output_function)
            w_list[i].solve()
            d_list[i].interpolate(v_list[i].function * w_list[i].output_function)
            mu.interpolate(mu * (d_list[i] ** alphas[i]))

        '''
        # Normalise mu
        Im_mu = assemble(mu * dx)
        mu.interpolate(mu / Im_mu)
        '''

        for i in range(num_dists):
            v_list[i].update(v_list[i].function * (mu / d_list[i]))
        print("Barycenter computation complete.")

    return mu

V = FunctionSpace(UnitSquareMesh(10, 10), "CG", 1)

mean_0 = [0.25, 0.25]
mean_1 = [0.75, 0.75]
mean_2 = [0.25, 0.75]

sigma_0 = 0.1
sigma_1 = 0.1
sigma_2 = 0.1

mu_0 = Function(V)
mu_1 = Function(V)
mu_2 = Function(V)

x, y = SpatialCoordinate(V.mesh())
mu_0.interpolate((1 / (2 * pi * sigma_0**2)) * exp(-((x- mean_0[0])**2 + (y - mean_0[1])**2) / (2 * sigma_0**2)))
mu_1.interpolate((1 / (2 * pi * sigma_1**2)) * exp(-((x- mean_1[0])**2 + (y - mean_1[1])**2) / (2 * sigma_1**2)))
mu_2.interpolate((1 / (2 * pi * sigma_2**2)) * exp(-((x- mean_2[0])**2 + (y - mean_2[1])**2) / (2 * sigma_2**2)))

# Normalise on mesh
Imu_0 = assemble(mu_0*dx)
Imu_1 = assemble(mu_1*dx)
Imu_2 = assemble(mu_2*dx)
mu_1.assign(mu_1/Imu_1)
mu_0.assign(mu_0/Imu_0)
mu_2.assign(mu_2/Imu_2)

mus = [mu_0, mu_1, mu_2]
alphas = [0.6, 0.3, 0.1]

bary = wasserstein_barycenter(mus, alphas)
fig, axes = plt.subplots()
colors = tripcolor(bary, axes=axes)
fig.colorbar(colors)
plt.show()
