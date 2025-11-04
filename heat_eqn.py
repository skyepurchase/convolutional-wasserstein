from firedrake import *
from helpers import *

T = 1
N = 1
n = 30
mesh = UnitSquareMesh(n, n)
unp1 = heat_kernel(T, N, mesh)

print(unp1.dat.data)

