from firedrake import FunctionSpace, UnitSquareMesh

from utils import generate_gaussian, initialise_env, visualise_2D_transport
from sinkhorn import sinkhorn


VANILLA_EPS = [0.01]
HIER_EPS = [0.1, 0.05, 0.02, 0.01]

MEAN_0 = [0.1, 0.1]
MEAN_1 = [0.5, 0.5]
SIGMA_0 = SIGMA_1 = 0.1

NUM_ITERS = 100
NUM_LEVELS = 4


if __name__=='__main__':
# Create a single and hierarchical mesh
    vanilla = initialise_env(256, 1, UnitSquareMesh, FunctionSpace)
    hierarchical = initialise_env(256, NUM_LEVELS, UnitSquareMesh, FunctionSpace)

    # Initialise the distributions on the meshes
    vanilla_mu_0 = generate_gaussian(vanilla[0], MEAN_0, SIGMA_0)
    vanilla_mu_1 = generate_gaussian(vanilla[0], MEAN_1, SIGMA_1)
    hierarchical_mu_0 = generate_gaussian(hierarchical[-1], MEAN_0, SIGMA_0)
    hierarchical_mu_1 = generate_gaussian(hierarchical[-1], MEAN_1, SIGMA_1)

    print(f"RUNNING VANILLA FOR {NUM_LEVELS * NUM_ITERS} ITERATIONS\n")
    phi, _ = sinkhorn(
        vanilla_mu_0,
        vanilla_mu_1,
        vanilla,
        epsilons=VANILLA_EPS,
        maxiter=NUM_LEVELS*NUM_ITERS
    )
    visualise_2D_transport(
        vanilla[-1], phi, "vanilla",
        source=MEAN_0,
        target=MEAN_1
    )

    print(f"\nRUNNING HIERACHICAL FOR {NUM_ITERS} ITERATIONS PER LEVEL\n")
    phi, _ = sinkhorn(
        hierarchical_mu_0,
        hierarchical_mu_1,
        hierarchical,
        epsilons=HIER_EPS,
        maxiter=NUM_ITERS
    )
    visualise_2D_transport(
        hierarchical[-1], phi, "hierarchical",
        source=MEAN_0,
        target=MEAN_1
    )
