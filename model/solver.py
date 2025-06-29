from .base_optimize import SLS_Solver, SLS_SolverOptimizer
from .optimizer import SLS_SolOpt_VariableReduction
import cvxpy as cp
import time

class Pred_SLS_Sol_CVX(SLS_Solver):
    def __init__(self, optimizers=[SLS_SolOpt_VariableReduction], **options):
        SLS_Solver.__init__(self, None, optimizers, **options)
        self._sls_problem = None  # cp.Problem(cp.Minimize(0))
        self._solver_optimizers = []
        for sol_opt in optimizers:
            if issubclass(sol_opt, SLS_SolverOptimizer):
                self._solver_optimizers.append(sol_opt)

    def get_SLS_Problem(self):
        return self._sls_problem

    def solve(
            self,
            objective_value,
            constraints
    ):
        # time_start = time.perf_counter()
        # self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        # self._sls_problem.solve()
        # time_end   = time.perf_counter()
        # print(" without optimization %.8f, " % (time_end-time_start))

        time_start = time.perf_counter()
        for sol_opt in self._solver_optimizers:
            # apply the optimizers
            solver_status, objective_value, constraints = sol_opt.optimize(objective_value, constraints)
            if solver_status == 'infeasible':
                return 0.0, solver_status
        self._sls_problem = cp.Problem(cp.Minimize(objective_value), constraints)

        try:
            self._sls_problem.solve(**self._options)
        except cp.error.SolverError as err:
            self.errorMessage('SLS solver error, synthesis fails.\nError message: %s' % err)
            exit()
        except Exception as err:
            self.errorMessage('Synthesis fails.\nError message: %s' % err)
            exit()

        for sol_opt in self._solver_optimizers:
            # optimizers post-process
            sol_opt.postProcess()
        time_end = time.perf_counter()
        print(" with optimization %.8f, " % (time_end - time_start))

        problem_value = self._sls_problem.value
        solver_status = self._sls_problem.status

        return problem_value, solver_status


# class Pred_SLS_Decomp_Sol_CVX(SLS_Solver):
#     def __init__(self, optimizers=[SLS_SolOpt_VariableReduction], **options):
#         SLS_Solver.__init__(self, None, optimizers, **options)
#         self._sls_problem = None  # cp.Problem(cp.Minimize(0))
#         self._solver_optimizers = []
#         for sol_opt in optimizers:
#             if issubclass(sol_opt, SLS_SolverOptimizer):
#                 self._solver_optimizers.append(sol_opt)
#
#     def get_SLS_Problem(self):
#         return self._sls_problem
#
#     def solve(
#             self,
#             objective_value,
#             constraints
#     ):
#         # time_start = time.perf_counter()
#         # self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
#         # self._sls_problem.solve()
#         # time_end   = time.perf_counter()
#         # print(" without optimization %.8f, " % (time_end-time_start))
#
#         time_start = time.perf_counter()
#         for sol_opt in self._solver_optimizers:
#             solver_status, objective_value, constraints = sol_opt.optimize(objective_value, constraints)
#             if solver_status == 'infeasible':
#                 return 0.0, solver_status
#
#         if self._optimization_direction < 0:
#             self._sls_problem = cp.Problem(cp.Minimize(objective_value), constraints)
#         else:
#             self._sls_problem = cp.Problem(cp.Maximize(objective_value), constraints)
#
#         try:
#             self._sls_problem.solve(**self._options)
#         except cp.error.SolverError as err:
#             self.errorMessage('SLS solver error, synthesis fails.\nError message: %s' % err)
#             exit()
#         except Exception as err:
#             self.errorMessage('Synthesis fails.\nError message: %s' % err)
#             exit()
#
#         for sol_opt in self._solver_optimizers:
#             sol_opt.postProcess()
#         time_end = time.perf_counter()
#         print(" with optimization %.8f, " % (time_end - time_start))
#
#         problem_value = self._sls_problem.value
#         solver_status = self._sls_problem.status
#
#         return problem_value, solver_status
#
#     def decomposition(
#             self,
#             objective_values,
#             constraints
#     ):