from firedrake import *
from sinkhorn import *

def wasserstein_barycenter(mus, alphas, Vs, epsilons):
    """
    Compute the Wasserstein barycenter of given distributions.
    """

    num_dists = len(mus) # number of input distributions
    try:
        assert abs(sum(alphas)) == 1, "Weights must sum to 1."
    except AssertionError as e:
        print("Error in weights: ", e)
        raise e
    
    epsilon = 0.1
    V = FunctionSpace(UnitSquareMesh(32, 32), "CG", 1)

    mu = Function(V).assign(1.0)

    v_list = [HeatEquationSolver(V, dt=epsilon/2)] * num_dists
    w_list = [HeatEquationSolver(V, dt=epsilon/2)] * num_dists
    d_list = [Function(V).assign(1.0)] * num_dists

    for i in range(num_dists):
        v_list[i].initialise()
        w_list[i].initialise()
    
    # Placeholder for barycenter computation logic

    # THIS LOOP CAN BE PARALLELISED
    for i in range(num_dists):
        v_list[i].solve()
        w_list[i].update(mu / v_list[i].output_function)
        w_list[i].solve()
        d_list[i].interpolate(v_list[i].function * w_list[i].output_function)

        mu.interpolate(mu * (d_list[i] ** alphas[i]))

    
    pass


    



