import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM=True
except:
    print("WARNING: Could not import `tqdm` please run `pip install tqdm` within the virtual environment.")
    print("INFO: Running without progressbar\n")
    TQDM=False

from firedrake import *

from utils import initialise_env, generate_gaussian, visualise_2D_transport
from solvers import HeatEquationSolver


def sinkhorn(
    mu_0,
    mu_1,
    Vs,
    epsilons=[0.5, 0.1, 0.05, 0.01],
    tol=1e-6,
    maxiter=10
):
    """
    Repeat the sinkhorn iteration until the tolerance is reached or maximum iterations.

    Parameters
    ----------
    mu_0     : The source distribution defined on Vs[-1]
    mu_1     : The target distribution defined on Vs[-1]
    Vs       : The hierarchy of function spaces
    epsilons : The regularisation parameter schedule
    tol      : The tolerance at which to stop
    maxiter  : The maximum number of iterations
    """
    print("\nRunning Sinkhorn iteration...")

    phi = Function(Vs[-1])
    psi = Function(Vs[-1])
    curr_mu_0 = Function(Vs[0])
    curr_mu_1 = Function(Vs[0])

    Solver_0 = HeatEquationSolver(Vs[0], dt=epsilons[0]/2)
    Solver_1 = HeatEquationSolver(Vs[0], dt=epsilons[0]/2)

    Solver_1.initialise()

    if TQDM:
        outer_iter = tqdm(
            zip(Vs, epsilons),
            total=len(Vs),
            leave=False
        )
        inner_iter = tqdm(
            range(maxiter),
            leave=False
        )
    else:
        outer_iter = zip(Vs, epsilons)
        inner_iter = range(maxiter)

    final_res = float("inf")
    for V, eps in outer_iter:
        if TQDM:
            outer_iter.set_description(f"epsilon={eps}")
        else:
            print(f"\nRunning with epsilon={eps}\n")

        Solver_0.refine(V, eps/2)
        Solver_1.refine(V, eps/2)

        curr_mu_0 = assemble(interpolate(mu_0, V))
        curr_mu_1 = assemble(interpolate(mu_1, V))

        res = float("inf")
        for _ in inner_iter:
            # Stop iterating once tolerance is reached
            if tol > res: break

            Solver_1.solve()
            res = norm(
                Solver_0.function -
                (curr_mu_0 / Solver_1.output_function)
            )
            Solver_0.update(curr_mu_0 / Solver_1.output_function)
            Solver_0.solve()
            Solver_1.update(curr_mu_1 / Solver_0.output_function)

            if TQDM:
                inner_iter.set_description(f"residual={res:.6f}")
            else:
                print(res)

            final_res = res

    phi.interpolate(epsilons[-1] * ln(Solver_0.function))
    psi.interpolate(epsilons[-1] * ln(Solver_1.function))

    print(f"Done! Final residual: {final_res}\n")
    return phi, psi


if __name__=='__main__':
    from firedrake.pyplot import tripcolor

    epsilons = [0.1, 0.05, 0.02, 0.01]
    mean_0 = [0.1, 0.1]
    mean_1 = [0.5, 0.5]
    sigma_0 = 0.01
    sigma_1 = 0.1

    # Initialise mesh and function space
    Vs = initialise_env(2048, len(epsilons), UnitSquareMesh, FunctionSpace)

    # Generate Gaussian distributions
    mu_0 = generate_gaussian(Vs[-1], mean_0, sigma_0)
    mu_1 = generate_gaussian(Vs[-1], mean_1, sigma_1)

    phi, psi = sinkhorn(mu_0, mu_1, Vs, epsilons=epsilons)

    print("\nVisualising...")
    visualise_2D_transport(Vs[-1], phi, "test.pvd")
    print("\nSaved.")
