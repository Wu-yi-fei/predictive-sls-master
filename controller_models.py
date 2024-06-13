from base_process import ControllerModel
from noise_models import FixedNoiseVector
import numpy as np
import control as ctrl

'''
To create a customized controller model, inherit the base function and customize the specified methods.

class ControllerModel:
    def initialize (self):
        # initialize internal state
    def controlConvergence(self, y, **kwargs):
        # getControl without changing the internal state
        return u
    def getControl(self, y, **kwargs):
        return u
'''


class OpenLoop_Controller(ControllerModel):
    '''
    The controller that gives zero control signal.
    '''

    def __init__(self, n_u=0):
        self._n_u = n_u

    def initialize(self):
        self._u = np.zeros([self._n_u, 1])

    def controlConvergence(self, y, **kwargs):
        return self._u.copy()

    def getControl(self, y, **kwargs):
        return self._u.copy()


class Causal_StateFeedback_Controller(ControllerModel):
    def __init__(self, A, B, Q, R, _horizon=1):

        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

        self._horizon = _horizon

        self._n_x = np.shape(self._A)[1]
        self._n_u = np.shape(self._B)[1]

        self._x = np.zeros([self._n_x, 1])

        self._P, _, _ = ctrl.dare(self._A, self._B, self._Q, self._R)
        self._D = self._get_D(self._B, self._P, self._R)

    @staticmethod
    def _get_D(B, P, R):
        D = np.matmul(np.linalg.inv(R + np.matmul(np.matmul(np.transpose(B), P), B)), np.transpose(B))
        return D

    @staticmethod
    def _remove(array):  # e.g., [w0, w1,w2] -> [w1, w2]
        return array[1:]

    def initialize(self):
        self._x = np.zeros([self._n_x, 1])

        P = self._P
        D = self._D

        _E = np.matmul(P, np.matmul(self._A, self._x))
        print('K_0=',-np.matmul(D, self._A))
        self._u = -np.matmul(D, _E)

    def controlConvergence(self, x, **kwargs):
        return self._u.copy()

    def getControl(self, y, **kwargs):
        self._x = y
        self._horizon = self._horizon - 1
        P = self._P
        D = self._D
        _E = np.matmul(P, np.matmul(self._A, self._x))
        self._u = -np.matmul(D, _E)

        return self._u.copy()

class Noncausal_StateFeedback_Controller(ControllerModel):
    def __init__(self, A, B, Q, R, _horizon=1):

        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

        self._horizon = _horizon

        self._n_x = np.shape(self._A)[1]
        self._n_u = np.shape(self._B)[1]

        self._x = np.zeros([self._n_x, 1])
        self.hat_ws = np.zeros([self._horizon, self._n_x, 1])

        self._P, _, _ = ctrl.dare(self._A, self._B, self._Q, self._R)
        self._D = self._get_D(self._B, self._P, self._R)
        self._H = self._get_H(self._B, self._D)
        self._F = self._get_F(self._A, self._P, self._H)

    @staticmethod
    def _get_D(B, P, R):
        D = np.matmul(np.linalg.inv(R + np.matmul(np.matmul(np.transpose(B), P), B)), np.transpose(B))
        return D

    @staticmethod
    def _get_H(B, D):
        H = np.matmul(B, D)
        return H

    @staticmethod
    def _get_F(A, P, H):
        F = A - np.matmul(H, np.matmul(P, A))
        return F

    @staticmethod
    def _remove(array):  # e.g., [w0, w1,w2] -> [w1, w2]
        return array[1:]

    def initialize(self, hat_ws):
        self._x = np.zeros([self._n_x, 1])
        self.hat_ws = hat_ws

        P = self._P
        D = self._D
        F = self._F

        _G = 0
        _G1 = 0
        for s in range(0, self._horizon):
            _G += np.matmul(np.linalg.matrix_power(np.transpose(F), s), np.matmul(P, self.hat_ws[s]))
        for s in range(0, self._horizon):
            _G1 += np.matmul(np.linalg.matrix_power(np.transpose(F), s), P)
        _E = np.matmul(P, np.matmul(self._A, self._x))
        print('K_0',-np.matmul(D, self._A))
        print('L_0',-np.matmul(D, _G1))
        self._u = -np.matmul(D, _E) - np.matmul(D, _G)

    def controlConvergence(self, x, **kwargs):
        return self._u.copy()

    def getControl(self, y, **kwargs):
        self._x = y
        self._horizon = self._horizon - 1
        self.hat_ws = self._remove(self.hat_ws)
        P = self._P
        D = self._D
        F = self._F

        _G = np.zeros(np.shape(self._x))
        for s in range(0, self._horizon):
            _G += np.matmul(np.linalg.matrix_power(np.transpose(F), s), np.matmul(P, self.hat_ws[s]))
        _E = np.matmul(P, np.matmul(self._A, self._x))
        # print(np.shape(_G))
        # print(np.shape(D))
        # print(np.shape(_E))
        self._u = -np.matmul(D, _E) - np.matmul(D, _G)

        return self._u.copy()


class SLS_StateFeedback_Controller(ControllerModel):
    def __init__(self, n_x=0, n_u=0, FIR_horizon=1):
        self._n_x = n_x
        self._n_u = n_u
        self._fir_horizon = FIR_horizon

        self._Phi_x = []  # = [ Phi_x[0], Phi_x[1], Phi_x[2], ... Phi_x[FIR_horizon] ]
        self._Phi_u = []  # = [ Phi_x[0], Phi_u[1], Phi_u[2], ... Phi_x[FIR_horizon] ]

        self._delta = []
        self._tilde_x = np.zeros([self._n_x, 1])
        self._u = np.zeros([self._n_u, 1])

    @staticmethod
    def _conv_computation(M, N, lb, ub, k):
        # perform computation sum_{tau = lb}^{ub-1} M[t]N[t-k]
        if (len(M) == 0) or (len(N) == 0):
            return 0
        conv = 0
        for t in range(lb, ub):
            if (t < len(M)) and (t - k < len(N)) and (t - k >= 0):
                conv += np.matmul(M[t], N[t - k])
        return conv

    @staticmethod
    def _FIFO_insert(fifo_list, element, max_size):  # e.g., [w2,w1,w0] -> [w3,w2,w1,w0] -> [w3,w2,w1]
        fifo_list.insert(0, element)
        # maintain delta length
        while len(fifo_list) > max_size:
            fifo_list.pop(-1)

    def initialize(self, delta0=None):
        self._delta = []
        if delta0 is not None:
            if isinstance(delta0, list):
                for delta in delta0:
                    self.add_valid_delta(delta)
            else:
                self.add_valid_delta(delta0)

        self._tilde_x = np.zeros([self._n_x, 1])

    def add_valid_delta(self, delta=None):
        # check if the disturbance is valid
        if (delta.shape[0] == self._n_x) and (delta.shape[1] == 1):
            self._delta.append(delta)

    def controlConvergence(self, y, **kwargs):
        tilde_delta = [y - self._tilde_x] + self._delta
        u = self._conv_computation(M=self._Phi_u, N=tilde_delta, lb=1, ub=self._fir_horizon + 1, k=1)
        return u

    def getControl(self, y, **kwargs):
        self._FIFO_insert(self._delta, y - self._tilde_x, self._fir_horizon + 1)

        u =self._conv_computation(M=self._Phi_u, N=self._delta, lb=1, ub=self._fir_horizon + 1, k=1)
        self._tilde_x = self._conv_computation(M=self._Phi_x, N=self._delta, lb=2, ub=self._fir_horizon + 1, k=2)

        return u


class Predictive_SLS_StateFeedback_Controller(ControllerModel):
    def __init__(self, n_x=0, n_u=0, FIR_horizon=1):
        self._n_x = n_x
        self._n_u = n_u
        self._fir_horizon = FIR_horizon

        self._Phi_x = []  # = [ Phi_x[0], Phi_x[1], Phi_x[2], ... Phi_x[FIR_horizon] ]
        self._Phi_u = []  # = [ Phi_x[0], Phi_u[1], Phi_u[2], ... Phi_x[FIR_horizon] ]
        self._Phi_hat_x = []  # = [ Phi_hat_x[FIR_horizon], ..., Phi_hat_x[-1] , Phi_hat_x[0], Phi_hat_x[1], ..., Phi_hat_x[FIR_horizon] ]
        self._Phi_hat_u = []  # = [ Phi_hat_u[FIR_horizon], ..., Phi_hat_u[-1] , Phi_hat_u[0], Phi_hat_u[1], ..., Phi_hat_u[FIR_horizon] ]

        self._delta = []
        self._hat_delta = []
        self._tilde_x = np.zeros([self._n_x, 1])
        self._u = np.zeros([self._n_u, 1])

    @staticmethod
    def _conv_computation(M, N, lb, ub, k):
        # perform computation sum_{tau = lb}^{ub-1} M[t]N[t-k]
        if (len(M) == 0) or (len(N) == 0):
            return 0
        conv = 0
        for t in range(lb, ub):
            if (t < len(M)) and (t - k < len(N)) and (t - k >= 0):
                conv += np.matmul(M[t], N[t - k])
        return conv

    @staticmethod
    def _FIFO_insert(fifo_list, element, max_size):   # e.g., [w2,w1,w0] -> [w3,w2,w1,w0] -> [w3,w2,w1]
        fifo_list.insert(0, element)
        while len(fifo_list) > max_size:
            fifo_list.pop(-1)

    @staticmethod
    def _FIFO_insert_inverse(fifo_list, element, max_size):   # e.g., [w2,w1,w0] -> [w3,w2,w1,w0] -> [w3,w2,w1]
        fifo_list.append(element)
        while len(fifo_list) > max_size:
            fifo_list.pop(0)

    def initialize(self, delta0=None, hat_delta0=None):
        self._delta = []
        if delta0 is not None:
            if isinstance(delta0, list):
                for delta in delta0:
                    self.add_valid_delta(delta)
            else:
                self.add_valid_delta(delta0)

        self._hat_delta = []
        if hat_delta0 is not None:
            if isinstance(hat_delta0, list):
                for hat_delta in hat_delta0:
                    self.add_valid_hat_delta(hat_delta)
            else:
                self.add_valid_hat_delta(hat_delta0)

        self._tilde_x = np.zeros([self._n_x, 1])

    def add_valid_delta(self, delta=None):
        # check if the disturbance is valid
        if (delta.shape[0] == self._n_x) and (delta.shape[1] == 1):
            self._delta.append(delta)

    def add_valid_hat_delta(self, hat_delta=None):
        # check if the prediction is valid
        if (hat_delta.shape[0] == self._n_x) and (hat_delta.shape[1] == 1):
            self._hat_delta.append(hat_delta)

    def controlConvergence(self, y, hat_delta=None):
        tilde_delta = [y - self._tilde_x] + self._delta
        u = (self._conv_computation(M=self._Phi_u, N=tilde_delta, lb=1, ub=self._fir_horizon + 1, k=1) +
             self._conv_computation(M=self._Phi_hat_u, N=self._hat_delta, lb=1, ub=2 * self._fir_horizon + 1, k=1))
        return u

    def getControl(self, y, hat_w=None):
        self._FIFO_insert(self._delta, y - self._tilde_x, self._fir_horizon + 1)
        self._FIFO_insert(self._hat_delta, hat_w, 2 * self._fir_horizon + 1)
        # print("w", self._delta)
        # # print("Phi", self._Phi_x)
        # print("hat_w", self._hat_delta)
        # print("hat_Phi", self._Phi_hat_x)
        u = (
                self._conv_computation(M=self._Phi_u, N=self._delta, lb=1, ub=self._fir_horizon + 1, k=1) +
             self._conv_computation(M=self._Phi_hat_u, N=self._hat_delta, lb=1, ub=2 * self._fir_horizon + 1, k=1)
        )

        self._tilde_x = (
            self._conv_computation(M=self._Phi_x, N=self._delta, lb=2, ub=self._fir_horizon + 1, k=2) +
             self._conv_computation(M=self._Phi_hat_x, N=self._hat_delta, lb=1, ub=2 * self._fir_horizon + 1, k=1)
        )
        return u
