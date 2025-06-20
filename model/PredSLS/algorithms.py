import numpy as np

from model.PredSLS.constraints import *
from model.PredSLS.objective import *
from model.system_models import *
from model.controller_models import *
from model.solver import Pred_SLS_Sol_CVX
from scipy import linalg


'''
Here is the main algorithm of predictive SLS
'''


class Predictive_SLS(Algorithm):
    '''
    Synthesizing the predictive controller using System Level Synthesis method.
    '''

    def __init__(self, system_model=None,
                 FIR_horizon=1,
                 state_feedback=True,
                 predictive=True,
                 solver=None,
                 objective=None,
                 communication_speed = 1,
                 d_locality = 2
                 ):
        self._fir_horizon = FIR_horizon
        self._state_feedback = state_feedback
        self._predictive = predictive
        self._system_model = None

        self.setSystemModel(system_model=system_model)

        self.resetObj_Cons()

        self.setSolver(solver=solver)

        self._sls_constraints = Pred_SLS_Cons(state_feedback=self._state_feedback, predictive=self._predictive)
        self._locality_constraints = dLocalized(comm_speed=communication_speed, d=d_locality)

        self._sls_objective = None

    def initializePhi(self):
        self._Phi_x = []
        self._Phi_u = []
        self._Phi_hat_x = []
        self._Phi_hat_u = []

        if self._system_model is None:
            return

        self._use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        n_x = self._system_model._n_x
        n_u = self._system_model._n_u

        if self._use_state_feedback_version:
            for t in range((self._fir_horizon + 1) * (self._fir_horizon + 1)):
                self._Phi_x.append(cp.Variable(shape=(n_x, n_x)))
                self._Phi_u.append(cp.Variable(shape=(n_u, n_x)))

            if self._predictive:
                for t in range((self._fir_horizon) * (self._fir_horizon)):
                    self._Phi_hat_x.append(cp.Variable(shape=(n_x, n_x)))
                    self._Phi_hat_u.append(cp.Variable(shape=(n_u, n_x)))
        else:
            pass

    def setSystemModel(self, system_model):
        if isinstance(system_model, SystemModel):
            self._system_model = system_model

        self.initializePhi()

        return self

    def setSolver(self,solver):
        # optimizer is embedded in the solver
        if not isinstance(solver, SLS_Solver):
            solver = None
        if solver is None:
            self._solver = Pred_SLS_Sol_CVX()
        else:
            self._solver = solver
        self._solver._sls = self

    def getSolver(self):
        return self._solver

    def addObj_Cons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives.append(obj_or_cons)
        return self

    def setObj_Cons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        return self

    def resetObj_Cons(self):
        self.resetObjectives()
        self.resetConstraints()

    def resetObjectives(self):
        self._objectives = []
        self._optimal_objective_value = float('inf')

    def resetConstraints(self):
        self._constraints = []

    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value

    def get_SLS_Problem (self):
        if isinstance(self._solver, Pred_SLS_Sol_CVX):
            return self._solver.get_SLS_Problem()
        else:
            return None

    # overload plus and less than or equal operators as syntactic sugars
    def __add__(self, obj_or_cons):
        return self.addObj_Cons(obj_or_cons)

    def __lshift__ (self, obj_or_cons_or_system):
        if isinstance(obj_or_cons_or_system,SystemModel):
            return self.setSystemModel(system_model=obj_or_cons_or_system)
        else:
            return self.setObj_Cons(obj_or_cons=obj_or_cons_or_system)

    def sanityCheck (self):
        # send error message
        if not self._state_feedback:
            return self.errorMessage('Only support state-feedback case for now.')
        if self._system_model is None:
            return self.errorMessage('The system is not yet assigned.')
        if not isinstance(self._system_model, LTI_System):
            return self.errorMessage('The system should be LTI_System.')
        if not isinstance(self._fir_horizon,int):
            return self.errorMessage('FIR horizon must be integer.')
        if self._fir_horizon < 1:
            return self.errorMessage('FIR horizon must be at least 1.')

        return True

    def synthesizedControllerModel(self):

        # variables used by both the state-feedback and output-feedback versions
        n_x = self._system_model._n_x
        n_u = self._system_model._n_u
        total = self._fir_horizon + 1

        if self._use_state_feedback_version != (self._state_feedback or self._system_model._state_feedback):
            self.initializePhi()

        if self._predictive:
            controller = Predictive_SLS_StateFeedback_Controller(
                n_x=n_x,
                n_u=n_u,
                FIR_horizon=self._fir_horizon
            )
        else:
            controller = SLS_StateFeedback_Controller(
                n_x=n_x,
                n_u=n_u,
                FIR_horizon=self._fir_horizon
            )

        # optimization objective
        objective_value = 0
        # objective_value = self._sls_objective.addObjectiveValue(sls=self, objective_value=objective_value)

        for obj in self._objectives:
            objective_value = obj.addObjectiveValue(
                sls=self,
                objective_value=objective_value
            )
        # add SLS main constraints
        self._sls_constraints._state_feedback = self._use_state_feedback_version
        constraints = self._sls_constraints.addConstraints(sls=self)
        constraints = self._locality_constraints.addConstraints(sls=self, constraints=constraints)

        # the constraints might also introduce additional terms at the objective
        for cons in self._constraints:
            objective_value = cons.addObjectiveValue(
                sls=self,
                objective_value=objective_value
            )
            constraints = cons.addConstraints(
                sls=self,
                constraints=constraints
            )
        problem_value, solver_status = self._solver.solve(
            objective_value=objective_value,
            constraints=constraints
        )

        if solver_status == 'infeasible':
            self.warningMessage('SLS problem infeasible')
            return None
        elif solver_status == 'unbounded':
            self.warningMessage('SLS problem unbounded')
            return None
        else:
            # save the solved problem for the users to examine if needed
            self._optimal_objective_value = problem_value
            if self._predictive:
                controller._Phi_x = [None] * total * total
                controller._Phi_u = [None] * total * total
                controller._Phi_hat_x = [None] * (total - 1) * (total - 1)
                controller._Phi_hat_u = [None] * (total - 1) * (total - 1)
                for t in range(total*total):
                    controller._Phi_x[t] = self._Phi_x[t].value
                    controller._Phi_u[t] = self._Phi_u[t].value
                for t in range((total - 1)*(total - 1)):
                    controller._Phi_hat_x[t] = self._Phi_hat_x[t].value
                    controller._Phi_hat_u[t] = self._Phi_hat_u[t].value
                L_0 = 0
                # print(controller._Phi_hat_u)
                for t in range(0 * (total - 1), 1 * (total - 1) - 1):
                    # print(controller._Phi_hat_u[t + 1])
                    # print(controller._Phi_hat_x[t + 1])
                    L_0 += controller._Phi_hat_u[t + 1] - np.matmul(
                    np.matmul(controller._Phi_u[1 * total + 1], np.linalg.inv(controller._Phi_x[1 * total +1])), controller._Phi_hat_x[t+1])
                # print("--------------- Reployed K and L -------------------")
                # print("K_0=", np.matmul(controller._Phi_u[1 * total + 1], np.linalg.inv(controller._Phi_x[1 * total+ 1])))
                # print("L_0=", - L_0)

                # print("Phi_x=", np.sum(np.sum(np.absolute(controller._Phi_x[21*6:21*7]), axis=1), axis=1))
                # print("_Phi_x=", controller._Phi_x[: 11])
                # np.save('Phi_x.npy', np.array(controller._Phi_x))
                # np.save('Phi_u.npy', np.array(controller._Phi_u))
                # np.save('Phi_hat_x.npy', np.array(controller._Phi_hat_x))
                # np.save('Phi_hat_u.npy', np.array(controller._Phi_hat_u))
                # print(controller._Phi_u[11 * 9: 11 * 4])
                # print(controller._Phi_hat_x[10 * 3: 10 * 4])
                # print()
                # print()
                # print("Phi_hat_x=", np.sum(np.sum(np.absolute(controller._Phi_hat_x[20*5:20*6]), axis=1), axis=1))

                # print("Phi_x=", controller._Phi_x[15])
                # # print("Phi_hat_x + Phi_x=", controller._Phi_hat_x + controller._Phi_x)
                # print("Phi_hat_u=", controller._Phi_hat_x[14])
                # print("----------------------------------------------------")
            else:
                controller._Phi_x = [None] * total* total
                controller._Phi_u = [None] * total* total
                for t in range(total* total):
                    controller._Phi_x[t] = self._Phi_x[t].value
                    controller._Phi_u[t] = self._Phi_u[t].value
                # print("--------------- Reployed K -------------------")
                # print("Phi_u=", controller._Phi_u)
                # print("Phi_x=", controller._Phi_x)
                # print("K_0=", np.matmul(controller._Phi_u[1 * total + 1], np.linalg.inv(controller._Phi_x[1 * total + 1])))
                # print("----------------------------------------------------")
            return controller

class Distributed_SLS_Predictive(Algorithm):
    '''
    Synthesizing the predictive controller using System Level Synthesis method.
    '''

    def __init__(self, system_model=None,
                 FIR_horizon=1,
                 state_feedback=True,
                 predictive=True,
                 solver=None,
                 objective=None,
                 d_locality=100,
                 agent=0,
                 ):
        self._fir_horizon = FIR_horizon
        self._state_feedback = state_feedback
        self._predictive = predictive
        self._system_model = None
        self._agent = agent

        self.setSystemModel(system_model=system_model)

        self.resetObj_Cons()

        self.setSolver(solver=solver)

        self._sls_constraints = Dist_SLS_Pred_Cons(state_feedback=self._state_feedback, predictive=self._predictive)
        self._locality_constraints = dLocalizedDist(d=d_locality)

        if objective == "H2":
            self._sls_objective = SLS_Obj_LQ(Q_sqrt=linalg.sqrtm(system_model._C1), R_sqrt=linalg.sqrtm(system_model._D12))
        else:
            self._sls_objective = None

    def initializePhi(self):
        self._phi_x = []
        self._phi_u = []
        self._phi_hat_x = []
        self._phi_hat_u = []

        if self._system_model is None:
            return

        self._use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        n_x = self._system_model._n_x
        n_u = self._system_model._n_u

        if self._use_state_feedback_version:
            for t in range((self._fir_horizon + 1) * (self._fir_horizon + 1)):
                self._phi_x.append(cp.Variable(shape=(n_x,1)))
                self._phi_u.append(cp.Variable(shape=(n_u,1)))

            if self._predictive:
                for t in range((self._fir_horizon) * (self._fir_horizon)):
                    self._phi_hat_x.append(cp.Variable(shape=(n_x,1)))
                    self._phi_hat_u.append(cp.Variable(shape=(n_u,1)))
        else:
            pass

    def setSystemModel(self, system_model):
        if isinstance(system_model, SystemModel):
            self._system_model = system_model

        self.initializePhi()

        return self

    def setSolver(self,solver):
        # optimizer is embedded in the solver
        if not isinstance(solver, SLS_Solver):
            solver = None
        if solver is None:
            self._solver = Pred_SLS_Sol_CVX()
        else:
            self._solver = solver
        self._solver._sls = self

    def getSolver(self):
        return self._solver

    def addObj_Cons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives.append(obj_or_cons)
        return self

    def setObj_Cons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        return self

    def resetObj_Cons(self):
        self.resetObjectives()
        self.resetConstraints()

    def resetObjectives(self):
        self._objectives = []
        self._optimal_objective_value = float('inf')

    def resetConstraints(self):
        self._constraints = []

    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value

    def get_SLS_Problem (self):
        if isinstance(self._solver, Pred_SLS_Sol_CVX):
            return self._solver.get_SLS_Problem()
        else:
            return None

    # overload plus and less than or equal operators as syntactic sugars
    def __add__(self, obj_or_cons):
        return self.addObj_Cons(obj_or_cons)

    def __lshift__ (self, obj_or_cons_or_system):
        if isinstance(obj_or_cons_or_system,SystemModel):
            return self.setSystemModel(system_model=obj_or_cons_or_system)
        else:
            return self.setObj_Cons(obj_or_cons=obj_or_cons_or_system)

    def sanityCheck (self):
        # send error message
        if not self._state_feedback:
            return self.errorMessage('Only support state-feedback case for now.')
        if self._system_model is None:
            return self.errorMessage('The system is not yet assigned.')
        if not isinstance(self._system_model, LTI_System):
            return self.errorMessage('The system should be LTI_System.')
        if not isinstance(self._fir_horizon,int):
            return self.errorMessage('FIR horizon must be integer.')
        if self._fir_horizon < 1:
            return self.errorMessage('FIR horizon must be at least 1.')

        return True

    def synthesizedControllerModel(self):

        n_x = self._system_model._n_x
        n_u = self._system_model._n_u
        total = self._fir_horizon + 1

        if self._use_state_feedback_version != (self._state_feedback or self._system_model._state_feedback):
            self.initializePhi()

        # optimization objective
        objective_value = 0
        # objective_value = self._sls_objective.addObjectiveValue(sls=self, objective_value=objective_value)

        for obj in self._objectives:
            objective_value = obj.addObjectiveValue(
                sls=self,
                objective_value=objective_value
            )
        # add SLS main constraints
        self._sls_constraints._state_feedback = self._use_state_feedback_version
        constraints = self._sls_constraints.addConstraints(sls=self)
        constraints = self._locality_constraints.addConstraints(sls=self, constraints=constraints)

        # the constraints might also introduce additional terms at the objective
        for cons in self._constraints:
            constraints = cons.addConstraints(
                sls=self,
                constraints=constraints
            )
        problem_value, solver_status = self._solver.solve(
            objective_value=objective_value,
            constraints=constraints
        )
        # _sls_problem = cp.Problem(cp.Minimize(objective_value), constraints)
        # _sls_problem.solve()
        # problem_value = _sls_problem.value
        # solver_status = _sls_problem.status

        if solver_status == 'infeasible':
            self.warningMessage('SLS problem infeasible')
            return None
        elif solver_status == 'unbounded':
            self.warningMessage('SLS problem unbounded')
            return None
        else:
            # save the solved problem for the users to examine if needed
            self._optimal_objective_value = problem_value
            _phi_x = [None] * total * total
            _phi_u = [None] * total * total
            _phi_hat_x = [None] * (total - 1) * (total - 1)
            _phi_hat_u = [None] * (total - 1) * (total - 1)
            if self._predictive:
                for t in range(total * total):
                    _phi_x[t] = self._phi_x[t].value
                    _phi_u[t] = self._phi_u[t].value
                for t in range((total - 1) * (total - 1)):
                    _phi_hat_x[t] = self._phi_hat_x[t].value
                    _phi_hat_u[t] = self._phi_hat_u[t].value
                return _phi_x, _phi_hat_x, _phi_u, _phi_hat_u, self._optimal_objective_value
            else:
                for t in range(total * total):
                    _phi_x[t] = self._phi_x[t].value
                    _phi_u[t] = self._phi_u[t].value

                return _phi_x, _phi_u, self._optimal_objective_value
