from control import zeros

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
        Phi_x = [[ Phi_x[0] 0          ...    0     ]
                 [ Phi_x[1] Phi_x[0]   ...    0     ]
                 [ ...      ...        ...    0     ]
                 [ Phi_x[T] Phi_x[T-1] ... Phi_x[0] ]]

        predictive state-feedback constriants:
        [ zI-A, -B2 ][ Phi_x Phi_hat_x ] = [ I 0 ]
                     [ Phi_u Phi_hat_u ]
        where:
        Phi_hat_x = [[ Phi_hat_x[0,0] Phi_hat_x[0,1]  ...  Phi_hat_x[0,T] ]
                     [ Phi_hat_x[1,0] Phi_hat_x[1,1]  ...  Phi_hat_x[1,T] ]
                     [      ...            ...                   ...      ]
                     [ Phi_hat_x[T,0] Phi_hat_x[T,0]  ...  Phi_hat_x[T,T] ]]
        '''

        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        n_x = sls._system_model._n_x
        n_u = sls._system_model._n_u

        if self._state_feedback:
            # sls constraints
            for t in range(sls._fir_horizon + 1):
                constraints += [(sls._system_model._A @ sls._Phi_x[t * (sls._fir_horizon + 1)] + sls._system_model._B2 @ sls._Phi_u[
                                                            t * (sls._fir_horizon + 1)]) == np.zeros([n_x, n_x])]
            for t in range(sls._fir_horizon + 1):
                for k in range(sls._fir_horizon + 1):
                    if t == 0:
                        constraints += [sls._Phi_x[t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, n_x])]
                        constraints += [sls._Phi_u[t * (sls._fir_horizon + 1) + k] == np.zeros([n_u, n_x])]
                    elif k == 0:
                        constraints += [(sls._system_model._A @ sls._Phi_x[
                            t * (sls._fir_horizon + 1)] + sls._system_model._B2 @ sls._Phi_u[
                                             t * (sls._fir_horizon + 1)]) == np.zeros([n_x, n_x])]
                    elif t < k:
                        constraints += [sls._Phi_x[t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, n_x])]
                        constraints += [sls._Phi_u[t * (sls._fir_horizon + 1) + k] == np.zeros([n_u, n_x])]
                    elif t == k:
                        constraints += [sls._Phi_x[t * (sls._fir_horizon + 1) + k] == np.eye(n_x)]
                    else:
                        constraints += [
                            sls._Phi_x[t * (sls._fir_horizon + 1) + k] == (
                                    sls._system_model._A @ sls._Phi_x[
                                t * (sls._fir_horizon + 1) + k + 1] + sls._system_model._B2 @ sls._Phi_u[
                                        t * (sls._fir_horizon + 1) + k + 1])
                        ]
                        constraints += [
                            sls._Phi_x[t * (sls._fir_horizon + 1) + k] == (
                                    sls._system_model._A @ sls._Phi_x[
                                (t-1) * (sls._fir_horizon + 1) + k] + sls._system_model._B2 @ sls._Phi_u[
                                (t-1) * (sls._fir_horizon + 1) + k])
                        ]

            # constraints += [sls._Phi_x[0] == np.zeros([n_x, n_x])]
            # constraints += [sls._Phi_u[0] == np.zeros([n_u, n_x])]
            # constraints += [sls._Phi_x[1] == np.eye(n_x)]
            # constraints += [
            #     (sls._system_model._A @ sls._Phi_x[sls._fir_horizon] +
            #      sls._system_model._B2 @ sls._Phi_u[sls._fir_horizon]) == np.zeros([n_x, n_x])
            # ]  # Phi_x[fir_horizon] = 0
            # for t in range(1, sls._fir_horizon):
            #     constraints += [
            #         sls._Phi_x[t + 1] == (sls._system_model._A @ sls._Phi_x[t] + sls._system_model._B2 @ sls._Phi_u[t])
            #     ]


            if self._predictive:
                for k in range(sls._fir_horizon):
                    constraints += [(sls._system_model._A @ sls._Phi_hat_x[
                        sls._fir_horizon * (sls._fir_horizon - 1) + k] + sls._system_model._B2 @ sls._Phi_hat_u[
                                         sls._fir_horizon * (sls._fir_horizon - 1) + k]) == np.zeros([n_x, n_x])]
                for t in range(sls._fir_horizon):
                    for k in range(sls._fir_horizon):
                        # constraints += [sls._Phi_hat_x[t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, n_x])]
                        # constraints += [sls._Phi_hat_u[t * (sls._fir_horizon + 1) + k] == np.zeros([n_u, n_x])]
                        # if t == 0:
                        #     constraints += [sls._Phi_hat_x[k] == np.zeros([n_x, n_x])]
                        #     constraints += [sls._Phi_hat_u[k] == np.zeros([n_u, n_x])]
                        # if k == 0:
                        #     constraints += [sls._Phi_hat_x[t * (sls._fir_horizon + 1)] == np.zeros([n_x, n_x])]
                        #     constraints += [sls._Phi_hat_u[t * (sls._fir_horizon + 1)] == np.zeros([n_u, n_x])]
                        if t == 0:
                            constraints += [sls._Phi_hat_x[t + k] == np.zeros([n_x, n_x])]
                        # elif t == sls._fir_horizon:
                        #     constraints += [(
                        #                     sls._system_model._A @ sls._Phi_hat_x[t * (sls._fir_horizon + 1) + k] + sls._system_model._B2 @ sls._Phi_hat_u[(t - 1) * (sls._fir_horizon + 1) + k])]
                        # else:
                        #     constraints += [
                        #             sls._Phi_hat_x[t * sls._fir_horizon + k] == (
                        #                     sls._system_model._A @ sls._Phi_hat_x[(t - 1) * sls._fir_horizon + k]
                        #                     + sls._system_model._B2 @ sls._Phi_hat_u[(t - 1) * sls._fir_horizon + k])
                        #         ]
                        else:
                            constraints += [
                                    sls._Phi_hat_x[t * sls._fir_horizon + k] == (
                                            sls._system_model._A @ sls._Phi_hat_x[(t - 1) * sls._fir_horizon + k]
                                            + sls._system_model._B2 @ sls._Phi_hat_u[(t - 1) * sls._fir_horizon + k])
                                ]

        return constraints

class Dist_SLS_Pred_Cons(SLS_Constraint):
    def __init__(self, state_feedback=True, predictive=True):
        super().__init__()
        self._state_feedback = state_feedback
        self._predictive = predictive

    def addConstraints(self, sls, constraints=None):

        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        n_x = sls._system_model._n_x
        n_u = sls._system_model._n_u

        if self._state_feedback:
            for t in range(sls._fir_horizon + 1):
                for k in range(sls._fir_horizon + 1):
                    # constraints += [sls._system_model.A_perp @ sls._phi_x[
                    #     t * (sls._fir_horizon + 1) + k] + sls._system_model.B_perp @ sls._phi_u[
                    #                     t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, 1])]
                    if t == 0:
                        constraints += [sls._phi_x[t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, 1])]
                        constraints += [sls._phi_u[t * (sls._fir_horizon + 1) + k] == np.zeros([n_u, 1])]
                    elif k == 0:
                        constraints += [(sls._system_model._A @ sls._phi_x[
                            t * (sls._fir_horizon + 1)] + sls._system_model._B2 @ sls._phi_u[
                                             t * (sls._fir_horizon + 1)]) == np.zeros([n_x, 1])]
                    elif t < k:
                        constraints += [sls._phi_x[t * (sls._fir_horizon + 1) + k] == np.zeros([n_x, 1])]
                        constraints += [sls._phi_u[t * (sls._fir_horizon + 1) + k] == np.zeros([n_u, 1])]
                    elif t == k:
                        e_i = np.zeros([n_x, 1])
                        e_i[sls._agent, 0] = 1
                        constraints += [sls._phi_x[t * (sls._fir_horizon + 1) + k] == e_i]
                    else:
                        constraints += [
                            sls._phi_x[t * (sls._fir_horizon + 1) + k] == (
                                    sls._system_model._A @ sls._phi_x[
                                (t-1) * (sls._fir_horizon + 1) + k] + sls._system_model._B2 @ sls._phi_u[
                                (t-1) * (sls._fir_horizon + 1) + k])
                        ]

            if self._predictive:
                for k in range(sls._fir_horizon):
                    constraints += [(sls._system_model._A @ sls._phi_hat_x[
                        sls._fir_horizon * (sls._fir_horizon - 1) + k] + sls._system_model._B2 @ sls._phi_hat_u[
                                         sls._fir_horizon * (sls._fir_horizon - 1) + k]) == np.zeros([n_x, 1])]
                for t in range(sls._fir_horizon):
                    for k in range(sls._fir_horizon):
                        if t == 0:
                            constraints += [sls._phi_hat_x[t + k] == np.zeros([n_x, 1])]

                        else:
                            # constraints += [sls._system_model.A_perp @ sls._phi_hat_x[
                            #     (t - 1) * sls._fir_horizon + k] + sls._system_model.B_perp @ sls._phi_hat_u[
                            #                     (t - 1) * sls._fir_horizon + k] == np.zeros([n_x, 1])]
                            constraints += [
                                    sls._phi_hat_x[t * sls._fir_horizon + k] == (
                                            sls._system_model._A @ sls._phi_hat_x[(t - 1) * sls._fir_horizon + k]
                                            + sls._system_model._B2 @ sls._phi_hat_u[(t - 1) * sls._fir_horizon + k])
                                ]

        return constraints


class dLocalized(SLS_Constraint):

    def __init__(self, base=None, act_delay=0, comm_speed=1, d=2):
        '''
        act_delay: actuation delay
        comm_speed: communication speed
        d: for d-localized
        '''
        super().__init__()
        self._act_delay = act_delay
        self._comm_speed = comm_speed
        self._d = d

    def addConstraints(self, sls, constraints):
        # localized constraints
        # get localized supports
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u
        Phi_hat_x = sls._Phi_hat_x
        Phi_hat_u = sls._Phi_hat_u

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d - 1) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T

        # adjacency matrix for available information
        info_adj = np.eye(sls._system_model._n_x) > 0
        transmission_time = -self._comm_speed * self._act_delay
        for t in range(1, sls._fir_horizon+1):
            # transmission_time += self._comm_speed
            # while transmission_time >= 1:
            #     transmission_time -= 1
            #     info_adj = np.dot(info_adj, commsAdj)

            support_x = localityR # np.logical_and(info_adj, localityR)
            support_u = np.dot(absB2T, support_x) > 0
            for k in range(1, sls._fir_horizon+1):

                # shutdown those not in the support
                for ix, iy in np.ndindex(support_x.shape):
                    if support_x[ix, iy] == False:
                        constraints += [Phi_x[t * (sls._fir_horizon + 1) + k][ix, iy] == 0]
                        if sls._predictive:
                                constraints += [Phi_hat_x[(t - 1) * sls._fir_horizon + k - 1][ix, iy] == 0]

                for ix, iy in np.ndindex(support_u.shape):
                    if support_u[ix, iy] == False:
                        constraints += [Phi_u[t * (sls._fir_horizon + 1) + k][ix, iy] == 0]
                        if sls._predictive:
                            constraints += [Phi_hat_u[(t - 1) * sls._fir_horizon + k - 1][ix, iy] == 0]

        return constraints

class dLocalizedDist(SLS_Constraint):

    def __init__(self, base=None, act_delay=0, d=2):
        '''
        act_delay: actuation delay
        comm_speed: communication speed
        d: for d-localized
        '''
        super().__init__()
        self._act_delay = act_delay
        self._d = d

    def addConstraints(self, sls, constraints):
        # localized constraints
        # get localized supports
        phi_x = sls._phi_x
        phi_u = sls._phi_u
        phi_hat_x = sls._phi_hat_x
        phi_hat_u = sls._phi_hat_u

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T

        # adjacency matrix for available information
        for t in range(sls._fir_horizon):
            support_x = localityR[sls._agent, np.newaxis].T
            support_u = np.dot(absB2T, support_x) > 0
            for k in range(sls._fir_horizon):

                # shutdown those not in the support
                for ix, iy in np.ndindex(support_x.shape):
                    if support_x[ix, iy] == False:
                        constraints += [phi_x[t * (sls._fir_horizon + 1) + k][ix, iy] == 0]
                        if sls._predictive:
                            constraints += [phi_hat_x[(t - 1) * sls._fir_horizon + k - 1][ix, iy] == 0]

                for ix, iy in np.ndindex(support_u.shape):
                    if support_u[ix, iy] == False:
                        constraints += [phi_u[t * (sls._fir_horizon + 1) + k][ix, iy] == 0]
                        if sls._predictive:
                            constraints += [phi_hat_u[(t - 1) * sls._fir_horizon + k - 1][ix, iy] == 0]

        return constraints