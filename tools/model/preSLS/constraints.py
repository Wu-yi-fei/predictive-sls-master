from model.base_optimize import SLS_Constraint
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
        where:
        Phi_x = [[Phi_x[0] 0          ... 0       ]
                 [Phi_x[1] Phi_x[0]   ... 0       ]
                 [...      ...        ... 0       ]
                 [Phi_x[T] Phi_x[T-1] ... Phi_x[0]]]

        predictive state-feedback constriants:
        [ zI-A, -B2 ][ Phi_x Phi_hat_x ] = [ I 0 ]
                     [ Phi_u Phi_hat_u ]
        where:
        Phi_hat_x = [[Phi_hat_x[0,0] Phi_hat_x[0,1]  ...  Phi_hat_x[0,T]]
                     [Phi_hat_x[1,0] Phi_hat_x[1,1]  ...  Phi_hat_x[1,T]]
                     [     ...            ...                   ...     ]
                     [Phi_hat_x[T,0] Phi_hat_x[T,0]  ...  Phi_hat_x[T,T]]]
        '''

        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        n_x = sls._system_model._n_x

        if self._state_feedback:
            # sls constraints
            constraints += [sls._Phi_x[0] == np.eye(n_x)]
            constraints += [
                (sls._system_model._A @ sls._Phi_x[sls._fir_horizon - 1] +
                 sls._system_model._B2 @ sls._Phi_u[sls._fir_horizon - 1]) == np.zeros([n_x, n_x])
            ]  # Phi_x[fir_horizon] = 0
            for t in range(sls._fir_horizon - 1):
                constraints += [
                    sls._Phi_x[t + 1] == (sls._system_model._A @ sls._Phi_x[t] + sls._system_model._B2 @ sls._Phi_u[t])
                ]

            if self._predictive:
                constraints += [
                    (sls._system_model._A @ sls._Phi_hat_x[sls._fir_horizon * sls._fir_horizon - 1] +
                     sls._system_model._B2 @ sls._Phi_hat_u[sls._fir_horizon * sls._fir_horizon - 1]) == np.zeros(
                        [n_x, n_x])
                ]
                for t in range(sls._fir_horizon):
                    for k in range(sls._fir_horizon):
                        if t == 0:
                            constraints += [sls._Phi_hat_x[k] == np.zeros([n_x, n_x])]
                        else:
                            constraints += [
                                    sls._Phi_hat_x[t * sls._fir_horizon + k] == (
                                            sls._system_model._A @ sls._Phi_hat_x[(t - 1) * sls._fir_horizon + k] + sls._system_model._B2 @ sls._Phi_hat_u[(t - 1) * sls._fir_horizon + k])
                                ]
        else:
            # TODO
            pass

        return constraints