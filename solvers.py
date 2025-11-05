from firedrake import *


class HeatEquationSolver:
    def __init__(
        self,
        V,
        dt=0.1,
        params={
            'ksp_type': 'preonly',
            'pc_type': 'lu'
        }
    ):
        """
        A heat equation solver that uses a single step of backward euler.
        This results in a modified Helmholtz equation which can be solved by Firedrake.

        Parameters
        ----------
        V      : The function space the heat equation is solved in
        dt     : The time step (only one timestep is completed)
        params : The firedrake solver parameters
        """
        self.dt = dt

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
        self.params = params
        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=params
        )

    def solve(self):
        """
        A wrapper for the Firedrake LinearVariationalSolver solve.

        Returns
        -------
        output_function : The resulting solved function
        """
        self.solver.solve()
        return self.output_function

    def initialise(self):
        """
        Initialise the function to all ones to prevent blow-up
        """
        self.function.assign(1.0)

    def update(self, value):
        """
        Set the new value for the initial value function.

        Parameters
        ----------
        value : The value to assign to the value function.
        """
        self.function.interpolate(value)

    def refine(self, new_V, new_dt):

        """
        Transfer the current potential into a new refined space

        Parameters
        ----------
        new_V  : The new function space
        new_dt : The new timestep
        """
        self.dt = new_dt

        # Create new functions in the space
        self.u = TrialFunction(new_V)
        self.v = TestFunction(new_V)
        self.output_function = Function(new_V)

        # Assign the current function to the new space
        self.function = assemble(interpolate(self.function, new_V))

        # Setup problem and solver
        a = (
            self.dt * inner(grad(self.u), grad(self.v)) +
            inner(self.u, self.v)
        ) * dx
        L = inner(self.function, self.v) * dx

        self.problem = LinearVariationalProblem(
            a, L, self.output_function
        )
        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=self.params
        )
