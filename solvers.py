from firedrake import *


class HeatEquationSolver:
    def __init__(
        self,
        V,
        epsilon=0.1,
        params={
            'ksp_type': 'preonly',
            'pc_type': 'lu'
        }
    ):
        self.dt = epsilon / 2
        self.u = TrialFunction(V)
        self.v = TestFunction(V)

        self.function = Function(V)
        self.output_function = Function(V)

        a = (
            self.dt * inner(grad(self.u), grad(self.v)) +
            inner(self.u, self.v)
        ) * dx
        L = inner(self.function, self.v) * dx

        self.problem = LinearVariationalProblem(
            a, L, self.output_function
        )
        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=params
        )

    def solve(self):
        """
        A wrapper for the Firedrake LinearVariationalSolver solve.

        Returns:
        output_function : The resulting solved function
        """
        self.solver.solve()
        return self.output_function

    def initialise(self):
        self.function.assign(1.0)

    def update(self, value):
        """
        Set the new value for the initial value function.

        Parameters:
        value : The value to assign to the value function.
        """
        self.function.interpolate(value)
