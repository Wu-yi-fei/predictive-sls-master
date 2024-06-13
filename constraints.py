from base_optimize import SLS_Constraint
import cvxpy as cp
import numpy as np


class Pred_SLS_Cons(SLS_Constraint):
    '''
    The descrete-time SLS constrains
    '''
    def __init__(self, state_feedback=True, predictive=True):
        super().__init__()
        self._state_feedback = state_feedback
        self._predictive = predictive

    def addConstraints(self, sls, constraints=None):
        '''
        state-feedback constraints:
        [ zI-A, -B2 ][ Phi_x ] = I
                     [ Phi_u ]

        predictive state-feedback constriants:
        [ zI-A, -B2 ][ Phi_x Phi_hat_x ] = [ I 0 ]
                     [ Phi_u Phi_hat_u ]

        '''
        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        n_x = sls._system_model._n_x
        n_u = sls._system_model._n_u

        if self._state_feedback:
            # sls constraints
            constraints += [sls._Phi_x[0] == np.zeros([n_x, n_x])]  # Phi_x[0] = 0
            constraints += [sls._Phi_u[0] == np.zeros([n_u, n_x])]  # Phi_u[0] = 0
            constraints += [sls._Phi_x[1] == np.eye(n_x)]  # Phi_x[1] = I
            constraints += [
                (sls._system_model._A @ sls._Phi_x[sls._fir_horizon] +
                 sls._system_model._B2 @ sls._Phi_u[sls._fir_horizon]) == np.zeros([n_x, n_x])
            ]  # Phi_x[fir_horizon] = 0
            for t in range(1, sls._fir_horizon):
                constraints += [
                    sls._Phi_x[t + 1] == (sls._system_model._A @ sls._Phi_x[t] + sls._system_model._B2 @ sls._Phi_u[t])
                ]

            if self._predictive:

                # Phi_x, Phi_u, Phi_hat_x, Phi_hat_u are in z^{-1} RH_{\inf}.
                constraints += [sls._Phi_hat_x[0] == np.zeros([n_x, n_x])]  # Phi_hat_x[0] = 0
                constraints += [sls._Phi_hat_u[0] == np.zeros([n_u, n_x])]  # Phi_hat_u[0] = 0
                constraints += [sls._Phi_hat_x[1] == np.zeros([n_x, n_x])]  # Phi_hat_x[1] = 0
                # constraints += [sls._Phi_hat_u[1] == -np.eye(n_x)]

                constraints += [
                    (sls._system_model._A @ sls._Phi_hat_x[2 * sls._fir_horizon] +
                     sls._system_model._B2 @ sls._Phi_hat_u[2 * sls._fir_horizon]) == np.zeros([n_x, n_x])
                ]  # Phi_hat_x[2fir_horizon] = 0

                for t in range(1, 2 * sls._fir_horizon + 1):
                    constraints += [
                        sls._Phi_hat_x[t - 1] == (sls._system_model._A @ sls._Phi_hat_x[t] +
                                                  sls._system_model._B2 @ sls._Phi_hat_u[t])
                    ]
        else:
            # TODO
            pass

        return constraints