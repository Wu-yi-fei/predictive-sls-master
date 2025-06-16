from control import zeros
from control.matlab import initial

from model.base_optimize import SLS_Constraint
import numpy as np

class Dist_SLS_Pred_Decomp_Cons(SLS_Constraint):
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

        e_i = np.zeros([n_x, 1])
        e_i[sls._agent, 0] = 1
        for t in range(sls._k):
            constraints += [sls._phi_x[t] == np.zeros([n_x, 1])]
            constraints += [sls._phi_u[t] == np.zeros([n_u, 1])]
        constraints += [sls._phi_x[sls._k] == e_i]
        for t in np.arange(sls._k, sls._horizon - 1):
            constraints += [
                sls._phi_x[t + 1] == (
                        sls._system_model._A @ sls._phi_x[t]
                        + sls._system_model._B2 @ sls._phi_u[t])
            ]
        constraints += [
            np.zeros([n_x, 1]) == (
                    sls._system_model._A @ sls._phi_x[sls._horizon-1]
                    + sls._system_model._B2 @ sls._phi_u[sls._horizon-1])
        ]
        constraints += [sls._phi_hat_x[0] == np.zeros([n_x, 1])]
        for t in range(sls._horizon - 1):
            # constraints += [
            #     sls._system_model.A_perp @ sls._phi_hat_x[t] + sls._system_model.B_perp @ sls._phi_hat_u[
            #         t] == np.zeros(
            #         [n_x, 1])]
            constraints += [
                sls._phi_hat_x[t + 1] == (
                        sls._system_model._A @ sls._phi_hat_x[t]
                        + sls._system_model._B2 @ sls._phi_hat_u[t])
            ]
        constraints += [
            np.zeros([n_x, 1]) == (
                    sls._system_model._A @ sls._phi_hat_x[sls._horizon - 1]
                    + sls._system_model._B2 @ sls._phi_hat_u[sls._horizon - 1])
        ]


        return constraints

class Dist_SLS_Pred_Decomp_Cons1(SLS_Constraint):
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
            e_i = np.zeros([n_x, 1])
            e_i[sls._agent, 0] = 1


        if sls._k == sls._horizon - 1:
            constraints += [sls._phi_x[sls._k] == e_i]
            constraints += [
                np.zeros([n_x, 1]) == (
                        sls._system_model._A @ sls._phi_x[sls._k]
                        + sls._system_model._B2 @ sls._phi_u[sls._k])
            ]
            constraints += [
                np.zeros([n_x, 1]) == (
                        sls._system_model._A @ sls._phi_hat_x[sls._k]
                        + sls._system_model._B2 @ sls._phi_hat_u[sls._k])
            ]
            for t in range(sls._k):
                # constraints += [
                #     sls._system_model.A_perp @ sls._phi_hat_x[t] + sls._system_model.B_perp @ sls._phi_hat_u[
                #         t] == np.zeros(
                #         [n_x, 1])]
                constraints += [sls._phi_x[t] == np.zeros([n_x, 1])]
                constraints += [sls._phi_u[t] == np.zeros([n_u, 1])]
                if t == 0:
                    constraints += [sls._phi_hat_x[0] == np.zeros([n_x, 1])]
                constraints += [
                    sls._phi_hat_x[t + 1] == (
                            sls._system_model._A @ sls._phi_hat_x[t]
                            + sls._system_model._B2 @ sls._phi_hat_u[t])
                ]
        else:
            for t in range(sls._k):
                # constraints += [
                #     sls._system_model.A_perp @ sls._phi_x[t] + sls._system_model.B_perp @ sls._phi_u[
                #         t] == np.zeros(
                #         [n_x, 1])]
                if t == 0:
                    constraints += [sls._phi_x[0] == np.zeros([n_x, 1])]
                if t == sls._k - 1:
                    constraints += [
                        sls._phi_x[t + 1] == (
                                sls._system_model._A @ sls._phi_x[t]
                                + sls._system_model._B2 @ sls._phi_u[t] + e_i)]
                else:
                    constraints += [
                    sls._phi_x[t + 1] == (
                            sls._system_model._A @ sls._phi_x[t]
                            + sls._system_model._B2 @ sls._phi_u[t])]

        return constraints


class Dist_SLS_Pred_Decomp_Cons2(SLS_Constraint):
    def __init__(self, state_feedback=True, predictive=True):
        super().__init__()
        self._state_feedback = state_feedback
        self._predictive = predictive

    def addConstraints(self, sls, initial_k, constraints=None):

        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        n_x = sls._system_model._n_x
        n_u = sls._system_model._n_u

        if self._state_feedback:
            e_i = np.zeros([n_x, 1])
            e_i[sls._agent, 0] = 1
            initial_k = np.array(initial_k)
            constraints += [sls._phi_x_1[0] == e_i]
            for i in range(sls._system_model._n_x):
                constraints += [sls._phi_hat_x_1[0][i, 0] == initial_k[i, 0]]
            for t in np.arange(sls._horizon - sls._k):
                if t == sls._horizon - sls._k - 1:
                    constraints += [
                        np.zeros([n_x, 1]) == (
                                sls._system_model._A @ sls._phi_hat_x_1[t]
                                + sls._system_model._B2 @ sls._phi_hat_u_1[t])
                    ]
                    constraints += [
                        np.zeros([n_x, 1]) == (
                                sls._system_model._A @ sls._phi_x_1[t]
                                + sls._system_model._B2 @ sls._phi_u_1[t])
                    ]
                else:
                    constraints += [
                        sls._phi_hat_x_1[t + 1] == (
                                sls._system_model._A @ sls._phi_hat_x_1[t]
                                + sls._system_model._B2 @ sls._phi_hat_u_1[t])
                    ]
                    constraints += [
                        sls._phi_x_1[t + 1] == (
                                sls._system_model._A @ sls._phi_x_1[t]
                                + sls._system_model._B2 @ sls._phi_u_1[t])
                    ]
                # constraints += [sls._system_model._B2 @ sls._phi_hat_u_1[0] == 0]
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

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T
        support_x = localityR[sls._agent, np.newaxis].T
        support_u = np.dot(absB2T, support_x) > 0
        if sls._k < sls._horizon - 1:
            for ix, iy in np.ndindex(support_x.shape):
                if support_x[ix, iy] == False:
                    constraints += [phi_x[sls._k][ix, iy] == 0]
                    # constraints += [phi_hat_x[sls._k][ix, iy] == 0]
            for t in range(sls._k):
                # shutdown those not in the support
                for ix, iy in np.ndindex(support_x.shape):
                    if support_x[ix, iy] == False:
                        constraints += [phi_x[t][ix, iy] == 0]
                        # constraints += [phi_hat_x[t][ix, iy] == 0]

                for ix, iy in np.ndindex(support_u.shape):
                    if support_u[ix, iy] == False:
                        constraints += [phi_u[t][ix, iy] == 0]
                        # constraints += [phi_hat_u[t][ix, iy] == 0]
        else:
            phi_hat_x = sls._phi_hat_x
            phi_hat_u = sls._phi_hat_u
            for t in range(sls._k+1):
                # shutdown those not in the support
                for ix, iy in np.ndindex(support_x.shape):
                    if support_x[ix, iy] == False:
                        constraints += [phi_x[t][ix, iy] == 0]
                        constraints += [phi_hat_x[t][ix, iy] == 0]

                for ix, iy in np.ndindex(support_u.shape):
                    if support_u[ix, iy] == False:
                        constraints += [phi_u[t][ix, iy] == 0]
                        constraints += [phi_hat_u[t][ix, iy] == 0]

        # adjacency matrix for available information

        return constraints


class dLocalizedDist2(SLS_Constraint):

    def __init__(self, base=None, act_delay=0, d=2):
        '''
        act_delay: actuation delay
        comm_speed: communication speed
        d: for d-localized
        '''
        super().__init__()
        self._act_delay = act_delay
        self._d = d

    def addConstraints(self, sls, initial_k, constraints):
        # localized constraints
        # get localized supports
        phi_x = sls._phi_x_1
        phi_u = sls._phi_u_1
        phi_hat_x = sls._phi_hat_x_1
        phi_hat_u = sls._phi_hat_u_1

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T

        support_x = localityR[sls._agent, np.newaxis].T
        support_u = np.dot(absB2T, support_x) > 0
        initial_k = np.array(initial_k)
        for i in range(sls._system_model._n_x):
            if initial_k[i, 0] != 0:
                constraints += [phi_hat_x[0][i, 0] == initial_k[i, 0]]
        for ix, iy in np.ndindex(support_x.shape):
            if support_x[ix, iy] == False:
                constraints += [phi_hat_x[0][ix, iy] == 0]
            # else:
            #     constraints += [phi_hat_x[0][ix, iy] == initial_k[ix, iy]]
        for ix, iy in np.ndindex(support_u.shape):
            if support_u[ix, iy] == False:
                constraints += [phi_u[0][ix, iy] == 0]
                constraints += [phi_hat_u[0][ix, iy] == 0]

        # adjacency matrix for available information
        for t in range(1, sls._horizon - sls._k):

            # shutdown those not in the support
            for ix, iy in np.ndindex(support_x.shape):
                if support_x[ix, iy] == False:
                    constraints += [phi_x[t][ix, iy] == 0]
                    constraints += [phi_hat_x[t][ix, iy] == 0]

            for ix, iy in np.ndindex(support_u.shape):
                if support_u[ix, iy] == False:
                    constraints += [phi_u[t][ix, iy] == 0]
                    constraints += [phi_hat_u[t][ix, iy] == 0]

        return constraints


class dLocalizedDist0(SLS_Constraint):

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
        support_x = localityR[sls._agent, np.newaxis].T
        support_u = np.dot(absB2T, support_x) > 0

        for t in range(sls._horizon - 1):
            # shutdown those not in the support
            for ix, iy in np.ndindex(support_x.shape):
                if support_x[ix, iy] == False:
                    constraints += [phi_x[t][ix, iy] == 0]
                    constraints += [phi_hat_x[t][ix, iy] == 0]

            for ix, iy in np.ndindex(support_u.shape):
                if support_u[ix, iy] == False:
                    constraints += [phi_u[t][ix, iy] == 0]
                    constraints += [phi_hat_u[t][ix, iy] == 0]

        # adjacency matrix for available information

        return constraints