import cvxpy as cp
import numpy as np
from cvxpy.utilities.power_tools import decompose

from model.decomposition.constraints import *
from model.decomposition.objective import *
from model.system_models import *
from model.controller_models import *
from model.solver import Pred_SLS_Sol_CVX
from model.base_process import *
from scipy import linalg

class Distributed_SLS_Predictive_Decomp(Algorithm):
    '''
    Synthesizing the predictive controller using System Level Synthesis method.
    '''

    def __init__(self, system_model=None,
                 horizon=1,
                 state_feedback=True,
                 predictive=True,
                 communication_speed = 1,
                 d_locality = 2,
                 agent=0,
                 k = 0,
                 decompose = True
                 ):
        self._horizon = horizon
        self._state_feedback = state_feedback
        self._predictive = predictive
        self._system_model = system_model
        self._agent = agent
        self._k = k
        self.decompose = decompose

        self.setSystemModel(system_model=system_model)

        self._solver = Pred_SLS_Sol_CVX()
        self._solver_1 = Pred_SLS_Sol_CVX()
        self._solver_2 = Pred_SLS_Sol_CVX()

        self._sls_constraints_step_1 = Dist_SLS_Pred_Decomp_Cons1(state_feedback=self._state_feedback,predictive=self._predictive)
        self._sls_constraints_step_2 = Dist_SLS_Pred_Decomp_Cons2(state_feedback=self._state_feedback,
                                                                  predictive=self._predictive)
        self._sls_constraints = Dist_SLS_Pred_Decomp_Cons(state_feedback=self._state_feedback,
                                                                  predictive=self._predictive)
        self._locality_constraints = dLocalizedDist(d=d_locality)
        self._locality_constraints_1 = dLocalizedDist2(d=d_locality)
        self._locality_constraints_2 = dLocalizedDist0(d=d_locality)

        self._sls_objective_step_1 = SLS_Obj_LQ_Dist_Decomp_1()
        self._sls_objective_step_2 = SLS_Obj_LQ_Dist_Decomp_2()
        self._sls_objective = SLS_Obj_LQ_Dist_Decomp()

        # self._locality_constraints = dLocalized(comm_speed=communication_speed, d=d_locality)

    def initialize_phi(self):
        self._phi_x = []
        self._phi_u = []
        self._phi_hat_x = []
        self._phi_hat_u = []

        self._phi_x_1 = []
        self._phi_u_1 = []
        self._phi_hat_x_1 = []
        self._phi_hat_u_1 = []

        if self._system_model is None:
            return

        n_x = self._system_model._n_x
        n_u = self._system_model._n_u
        if self.decompose:
            if self._k == 0:
                for t in range(self._horizon):
                    self._phi_x_1.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_u_1.append(cp.Variable(shape=(n_u, 1)))
                    self._phi_hat_x_1.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_hat_u_1.append(cp.Variable(shape=(n_u, 1)))
            elif self._k == self._horizon - 1:
                for t in range(self._horizon):
                    self._phi_x.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_u.append(cp.Variable(shape=(n_u, 1)))
                    self._phi_hat_x.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_hat_u.append(cp.Variable(shape=(n_u, 1)))
            else:
                for t in range(self._k):
                    self._phi_x.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_u.append(cp.Variable(shape=(n_u, 1)))
                self._phi_x.append(cp.Variable(shape=(n_x, 1)))
                for t in range(self._horizon - self._k):
                    self._phi_x_1.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_u_1.append(cp.Variable(shape=(n_u, 1)))
                    self._phi_hat_x_1.append(cp.Variable(shape=(n_x, 1)))
                    self._phi_hat_u_1.append(cp.Variable(shape=(n_u, 1)))
        else:
            for t in range(self._horizon):
                self._phi_x.append(cp.Variable(shape=(n_x, 1)))
                self._phi_u.append(cp.Variable(shape=(n_u, 1)))
                self._phi_hat_x.append(cp.Variable(shape=(n_x, 1)))
                self._phi_hat_u.append(cp.Variable(shape=(n_u, 1)))



    def process(self):

        # variables used by both the state-feedback and output-feedback versions
        n_x = self._system_model._n_x
        n_u = self._system_model._n_u
        self.initialize_phi()
        if self.decompose:
            if self._k == 0:
                objective_value_step_2 = 0
                objective_value_step_2 = self._sls_objective_step_2.addObjectiveValue(
                    sls=self,
                    objective_value=objective_value_step_2
                )
                constraints = self._sls_constraints_step_2.addConstraints(sls=self, initial_k=np.zeros(
                    [self._system_model._n_x, 1]))
                constraints = self._locality_constraints_1.addConstraints(sls=self, constraints=constraints,
                                                                          initial_k=np.zeros(
                                                                              [self._system_model._n_x, 1]))
                problem_value_step_2, solver_status_step_2 = self._solver_2.solve(
                    objective_value=objective_value_step_2,
                    constraints=constraints
                )

                if solver_status_step_2 == 'infeasible':
                    self.warningMessage('problem 2 infeasible')
                    return None
                elif solver_status_step_2 == 'unbounded':
                    self.warningMessage('problem 2 unbounded')
                    return None

                phi_x = [None] * self._horizon
                phi_u = [None] * self._horizon
                phi_hat_x = [None] * self._horizon
                phi_hat_u = [None] * self._horizon
                for t in range(self._horizon):
                    phi_x[t] = self._phi_x_1[t].value
                    phi_u[t] = self._phi_u_1[t].value
                    phi_hat_x[t] = self._phi_hat_x_1[t].value
                    phi_hat_u[t] = self._phi_hat_u_1[t].value

            elif self._k == self._horizon - 1:
                objective_value_step_1 = 0
                objective_value_step_1 = self._sls_objective_step_1.addObjectiveValue(
                    sls=self,
                    objective_value=objective_value_step_1
                )
                constraints = self._sls_constraints_step_1.addConstraints(sls=self)
                constraints = self._locality_constraints.addConstraints(sls=self, constraints=constraints)
                problem_value_step_1, solver_status_step_1 = self._solver_1.solve(
                    objective_value=objective_value_step_1,
                    constraints=constraints
                )

                if solver_status_step_1 == 'infeasible':
                    self.warningMessage('problem 1 infeasible')
                    return None
                elif solver_status_step_1 == 'unbounded':
                    self.warningMessage('problem 1 unbounded')
                    return None

                phi_x = [None] * self._horizon
                phi_u = [None] * self._horizon
                phi_hat_x = [None] * self._horizon
                phi_hat_u = [None] * self._horizon
                for t in range(self._horizon):
                    phi_x[t] = self._phi_x[t].value
                    phi_u[t] = self._phi_u[t].value
                    phi_hat_x[t] = self._phi_hat_x[t].value
                    phi_hat_u[t] = self._phi_hat_u[t].value


            else:
                objective_value_step_1 = 0

                objective_value_step_1 = self._sls_objective_step_1.addObjectiveValue(
                    sls=self,
                    objective_value=objective_value_step_1
                )
                constraints = self._sls_constraints_step_1.addConstraints(sls=self)
                constraints = self._locality_constraints.addConstraints(sls=self, constraints=constraints)
                problem_value_step_1, solver_status_step_1 = self._solver_1.solve(
                    objective_value=objective_value_step_1,
                    constraints=constraints
                )

                if solver_status_step_1 == 'infeasible':
                    self.warningMessage('problem 1 infeasible')
                    return None
                elif solver_status_step_1 == 'unbounded':
                    self.warningMessage('problem 1 unbounded')
                    return None

                e_i = np.zeros([n_x, 1])
                e_i[self._agent, 0] = 1
                initial = self._phi_x[self._k].value - e_i
                # print(initial)

                phi_x = [None] * self._horizon
                phi_u = [None] * self._horizon
                phi_hat_x = [None] * self._horizon
                phi_hat_u = [None] * self._horizon
                for t in range(self._k):
                    phi_x[t] = np.zeros([n_x, 1])
                    phi_u[t] = np.zeros([n_x, 1])
                    phi_hat_x[t] = self._phi_x[t].value
                    phi_hat_u[t] = self._phi_u[t].value

                objective_value_step_2 = 0
                objective_value_step_2 = self._sls_objective_step_2.addObjectiveValue(
                    sls=self,
                    objective_value=objective_value_step_2
                )

                constraints = self._sls_constraints_step_2.addConstraints(sls=self, initial_k=initial)
                constraints = self._locality_constraints_1.addConstraints(sls=self, constraints=constraints,
                                                                          initial_k=initial)
                problem_value_step_2, solver_status_step_2 = self._solver_2.solve(
                    objective_value=objective_value_step_2,
                    constraints=constraints
                )

                if solver_status_step_2 == 'infeasible':
                    self.warningMessage('problem 2 infeasible')
                    return None
                elif solver_status_step_2 == 'unbounded':
                    self.warningMessage('problem 2 unbounded')
                    return None

                for t in range(self._horizon - self._k):
                    phi_x[self._k + t] = self._phi_x_1[t].value
                    phi_u[self._k + t] = self._phi_u_1[t].value
                    phi_hat_x[self._k + t] = self._phi_hat_x_1[t].value
                    phi_hat_u[self._k + t] = self._phi_hat_u_1[t].value
        else:
            objective_value = 0

            objective_value = self._sls_objective.addObjectiveValue(
                sls=self,
                objective_value=objective_value
            )
            constraints = self._sls_constraints.addConstraints(sls=self)
            constraints = self._locality_constraints_2.addConstraints(sls=self, constraints=constraints)
            problem_value, solver_status = self._solver.solve(
                objective_value=objective_value,
                constraints=constraints
            )

            if solver_status == 'infeasible':
                self.warningMessage('problem infeasible')
                return None
            elif solver_status == 'unbounded':
                self.warningMessage('problem unbounded')
                return None

            phi_x = [None] * self._horizon
            phi_u = [None] * self._horizon
            phi_hat_x = [None] * self._horizon
            phi_hat_u = [None] * self._horizon
            for t in range(self._horizon):
                phi_x[t] = self._phi_x[t].value
                phi_u[t] = self._phi_x[t].value
                phi_hat_x[t] = self._phi_hat_x[t].value
                phi_hat_u[t] = self._phi_hat_u[t].value

        return phi_x, phi_u, phi_hat_x, phi_hat_u





