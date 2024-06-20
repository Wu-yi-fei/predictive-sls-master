from model.base_optimize import *
import numpy as np
import cvxpy as cp


class SLS_Obj_LQ(SLS_Objective):
    '''
    This function assumes
    1) all disturbance / noise is zero-centered Gaussian
    2) process disturbance and measurement noise are uncorrelated

    Cov_w is the covariance matrix for process disturbance
    Cov_v is the covariance matrix for measurement noise

    return
        || [Q^0.5, R^0.5][Phi_x; Phi_u]Cov_w^0.5 ||_Frob^2
        i.e. LQR for state-feedback SLS

        || [Q^0.5, R^0.5][Phi_xx, Phi_xy; Phi_ux, Phi_uy][Cov_w^0.5; Cov_v^0.5]||_Frob^2
        i.e. LQG for output-feedback SLS
    '''

    def __init__(self, Q_sqrt=None, R_sqrt=None, Cov_w_sqrt=None, Cov_hat_w_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt
        self._Cov_w_sqrt = Cov_w_sqrt
        self._Cov_hat_w_sqrt = Cov_hat_w_sqrt

    def addObjectiveValue(self, sls, objective_value):
        Q_sqrt = self._Q_sqrt
        R_sqrt = self._R_sqrt
        Cov_w_sqrt = self._Cov_w_sqrt
        Cov_hat_w_sqrt = self._Cov_hat_w_sqrt

        # default values
        if Q_sqrt is None:
            Q_sqrt = sls._system_model._C1
        if R_sqrt is None:
            R_sqrt = sls._system_model._D12
        if Cov_w_sqrt is None:
            Cov_w_sqrt = sls._system_model._B1
        if Cov_hat_w_sqrt is None:
            Cov_hat_w_sqrt = sls._system_model._B1

        self._objective_expression = 0
        if sls._predictive:
            # predictive feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u
            Phi_hat_x = sls._Phi_hat_x
            Phi_hat_u = sls._Phi_hat_u
            for t in range(len(Phi_x)):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ Phi_x[t] +
                    Q_sqrt @ Phi_hat_x[len(Phi_x) - t - 1]) + cp.sum_squares(
                    R_sqrt @ Phi_u[t] +
                    R_sqrt @ Phi_hat_u[len(Phi_x) - t - 1]
                )
            for t in range(len(Phi_hat_x) - len(Phi_x)):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ Phi_hat_x[len(Phi_x) + t]) + cp.sum_squares(
                    R_sqrt @ Phi_hat_u[len(Phi_x) + t]
                )
        else:
            # feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u

            for t in range(len(Phi_x)):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ Phi_x[t]) + cp.sum_squares(
                    R_sqrt @ Phi_u[t]
                )
        return objective_value + self._objective_expression

class SLS_Obj_H2(SLS_Obj_LQ):
    '''
    return
        || [C1, D12][Phi_x; Phi_u] ||_H2^2
        for state-feedback SLS
    '''

class LQC(SLS_Objective):

    def __init__(self, Q_sqrt=None, R_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt

    def ObjectiveValue(self, alg=None, us=None, xs=None):
        Q_sqrt = self._Q_sqrt
        R_sqrt = self._R_sqrt
        # default values
        if Q_sqrt is None:
            Q_sqrt = alg._system_model._C1
        if R_sqrt is None:
            R_sqrt = alg._system_model._D12

        self._objective_expression = 0
        # objective function
        for t in range(len(xs)):
            self._objective_expression += (np.matmul(np.transpose(R_sqrt @ us[t]), (R_sqrt @ us[t]))
                                           + np.matmul(np.transpose(Q_sqrt @ xs[t]), (Q_sqrt @ xs[t]))
                )
            # self._objective_expression += np.square(np.linalg.norm(
            #     Q_sqrt @ xs[t]
            # )) + np.square(np.linalg.norm(
            #     R_sqrt @ us[t]
            # ))
        return self._objective_expression