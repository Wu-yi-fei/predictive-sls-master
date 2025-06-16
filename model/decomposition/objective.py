from model.base_optimize import *
import numpy as np
import cvxpy as cp
import control as ctrl
from scipy.linalg import sqrtm

class SLS_Obj_LQ_Dist_Decomp_1(SLS_Objective):

    def __init__(self, Q_sqrt=None, R_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt

    def addObjectiveValue(self, sls, objective_value):
        Q_sqrt = self._Q_sqrt
        R_sqrt = self._R_sqrt

        # default values
        if Q_sqrt is None:
            Q_sqrt = sls._system_model._C1
        if R_sqrt is None:
            R_sqrt = sls._system_model._D12

        self._objective_expression = 0
        # print(_P)
        # predictive feedback
        phi_x = sls._phi_x
        phi_u = sls._phi_u
        phi_hat_x = sls._phi_hat_x
        phi_hat_u = sls._phi_hat_u
        if sls._k == sls._horizon - 1:
            for t in range(sls._horizon):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ (phi_x[t] + phi_hat_x[t])
                ) + cp.sum_squares(
                    R_sqrt @ (phi_u[t] + phi_hat_u[t])
                )
        else:
            _P, _, _ = ctrl.dare(sls._system_model._A, sls._system_model._B2, Q_sqrt, R_sqrt)
            _P = sqrtm(_P)
            for t in range(sls._k):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ (phi_x[t])
                ) + cp.sum_squares(
                    R_sqrt @ (phi_u[t])
                )
            self._objective_expression += cp.sum_squares(
                _P @ (phi_x[sls._k])
            )
        return objective_value + self._objective_expression


class SLS_Obj_LQ_Dist_Decomp_2(SLS_Objective):

    def __init__(self, Q_sqrt=None, R_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt

    def addObjectiveValue(self, sls, objective_value):
        Q_sqrt = self._Q_sqrt
        R_sqrt = self._R_sqrt

        # default values
        if Q_sqrt is None:
            Q_sqrt = sls._system_model._C1
        if R_sqrt is None:
            R_sqrt = sls._system_model._D12

        self._objective_expression = 0
        # predictive feedback
        phi_x = sls._phi_x_1
        phi_u = sls._phi_u_1
        phi_hat_x = sls._phi_hat_x_1
        phi_hat_u = sls._phi_hat_u_1
        for t in range(sls._horizon - sls._k):
            self._objective_expression += cp.sum_squares(
                    Q_sqrt @ (phi_x[t] + phi_hat_x[t])
            ) + cp.sum_squares(
                    R_sqrt @ (phi_u[t] + phi_hat_u[t])
            )
        return objective_value + self._objective_expression


class SLS_Obj_LQ_Dist_Decomp(SLS_Objective):

    def __init__(self, Q_sqrt=None, R_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt

    def addObjectiveValue(self, sls, objective_value):
        Q_sqrt = self._Q_sqrt
        R_sqrt = self._R_sqrt

        # default values
        if Q_sqrt is None:
            Q_sqrt = sls._system_model._C1
        if R_sqrt is None:
            R_sqrt = sls._system_model._D12

        self._objective_expression = 0
        # predictive feedback
        phi_x = sls._phi_x
        phi_u = sls._phi_u
        phi_hat_x = sls._phi_hat_x
        phi_hat_u = sls._phi_hat_u
        for t in range(sls._horizon):
            self._objective_expression += cp.sum_squares(
                    Q_sqrt @ (phi_x[t] + phi_hat_x[t])
            ) + cp.sum_squares(
                    R_sqrt @ (phi_u[t] + phi_hat_u[t])
            )
        return objective_value + self._objective_expression