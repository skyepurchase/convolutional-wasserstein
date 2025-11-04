from firedrake import *

#Time discretization
T = 1
N = 1
dt = Constant(T / N)

n = 30
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)

un = Function(V) #Initial condition (sol. at previous timestep)
un.assign(1.0) 

u = TrialFunction(V)
v = TestFunction(V)

a = (dt * inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(un, v) * dx

unp1 = Function(V) #Redefine to be a function holding the solution:

params = {'ksp_type':'preonly', 'pc_type':'lu'}
prob = LinearVariationalProblem(a, L, unp1)
solv = LinearVariationalSolver(prob, solver_parameters=params)

solv.solve()

print(unp1.dat.data)
