from firedrake import *
from solvers import HeatEquationSolver
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor

def wasserstein_barycenter(mus, alphas, V):
    """
    Compute the Wasserstein barycenter of given distributions.
    """

    num_dists = len(mus) # number of input distributions
    try:
        assert abs(sum(alphas)) == 1, "Weights must sum to 1."
    except AssertionError as e:
        print("Error in weights: ", e)
        raise e
    
    epsilon = 0.05

    mu = Function(V, name="mu").assign(1.0)
    Im_mu = assemble(mu * dx)
    mu.interpolate(mu / Im_mu)

    test_func = Function(V).assign(0.0)

    v_list = []
    w_list = []
    d_list = []

    for _ in range(num_dists):
        v_list.append(HeatEquationSolver(V, dt=epsilon/2))
        w_list.append(HeatEquationSolver(V, dt=epsilon/2))
        d_list.append(Function(V).assign(1.0))

    '''
    v_list = [HeatEquationSolver(V, dt=epsilon/2) for _ in range(num_dists)]
    w_list = [HeatEquationSolver(V, dt=epsilon/2) for _ in range(num_dists)]
    d_list = [Function(V).assign(1.0) for _ in range(num_dists)]
    '''

    for i in range(num_dists):
        v_list[i].initialise()
        w_list[i].initialise()
    
    # Placeholder for barycenter computation logic

    curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

    j = 0
    tol=1e-5
    res = 1
    maxiter = 100
    while (res > tol) and (j < maxiter):
    #for j in range(num_dists):
        mu.assign(1.0)
        # THIS LOOP CAN BE PARALLELISED
        test_func.assign(w_list[0].function)
        for i in range(num_dists):
            v_list[i].solve()
            w_list[i].update(curr[i] / v_list[i].output_function)
            w_list[i].solve()
            d_list[i].interpolate(v_list[i].function * w_list[i].output_function)
            mu.interpolate(mu * (d_list[i] ** alphas[i]))

        res = norm(test_func - w_list[0].function)

        for i in range(num_dists):
            v_list[i].update(v_list[i].function * (mu / d_list[i]))
        #print("Barycenter computation complete.")

        #res = norm(mu - old_mu)
        print(f"Iteration {j}, Residual: {res}")

        j += 1

    return mu

n = 100
V = FunctionSpace(UnitSquareMesh(n, n), "CG", 1)

mean_0 = [0.4, 0.4]
mean_1 = [0.6, 0.6]
mean_2 = [0.4, 0.6]

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

mus = [mu_0, mu_1]
alphas = [0.5, 0.5]

bary = wasserstein_barycenter(mus, alphas, V)

VTKFile("bary1.pvd").write(mu_0, mu_1, mu_2, bary)

#colors = tripcolor(bary, axes=axes)
#fig.colorbar(colors)
#plt.show()
