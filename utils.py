from firedrake import *
from firedrake.pyplot import tripcolor

import matplotlib.pyplot as plt


def initialise_env(
    target_size,
    levels,
    mesher,
    make_space
):
    """
    Initialise a function space hierarchy

    Parameters
    ----------
    target_size : The size of the finest grid
    levels      : The number of levels to create
    mesher      : The firedrake mesh maker
    make_space  : The function space maker

    Returns
    -------
    vs : The hierarchy of function spaces
    """
    print("\nInitialising environment (this may take a while)...")

    # Short cut for non-hierarchical mesh
    if levels == 1:
        mesh = mesher(target_size, target_size)
        V = make_space(mesh, "CG", 1)
        return [V]

    # Calculate the coarse size to refine (each cell becomes 4)
    start_size = int(target_size / (4**(levels-1)))

    mesh = mesher(start_size, start_size)
    hierarchy = MeshHierarchy(mesh, levels)

    vs = []

    for sub_mesh in hierarchy:
        vs.append(make_space(sub_mesh, "CG", 1))

    print("Done!\n")

    return vs

def generate_gaussian(V, mean, sigma):
    """
    Generate simple Gaussian distribution for OT. Handles conversion to Firedrake symbolic function.

    Parameters
    ----------
    V         : Funnction space to define probability density functions on.
    mean      : 2d array containing mean of mu.
    sigma     : Standard deviation of mu.
    """

    # Set up probability distribution
    mu = Function(V)

    x, y = SpatialCoordinate(V.mesh())
    mu.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean[0])**2 + (y - mean[1])**2) / (2 * sigma**2)))

    # Normalise on mesh
    Imu = assemble(mu*dx)
    mu.assign(mu/Imu)
    return mu


def visualise_2D_transport(V, phi, filename):
    mesh = V.mesh()
    x, y = SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()

    T = Function(Vc).interpolate(as_vector((x,y)) + grad(phi))

    print("\nVisualising...")
    fig, axes = plt.subplots()
    colors = tripcolor(T, axes=axes)
    fig.colorbar(colors)
    plt.savefig(f"{filename}.png", format="png")
