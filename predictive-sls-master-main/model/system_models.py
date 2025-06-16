import numpy as np
from .base_process import SystemModel


class LTI_System(SystemModel):
    """
    customize a LTI system
    """
    def __init__(self, n_x=0, n_w=0, n_u=0, n_y=0, n_z=0, **kwargs):
        SystemModel.__init__(self, **kwargs)

        if not isinstance(n_x, int): n_x = 0
        if not isinstance(n_w, int): n_w = 0
        if not isinstance(n_u, int): n_u = 0
        if not isinstance(n_y, int): n_y = 0
        if not isinstance(n_z, int): n_z = 0

        # vector dimensions of
        self._n_x = n_x  # state
        self._n_w = n_w  # noise
        self._n_u = n_u  # control
        self._n_z = n_z  # output
        self._n_y = n_y  # measurement

        # state: x(t+1)= A * x(t) + B1 * w(t) + B2 * u(t)
        self._A = np.zeros([n_x, n_x])
        self._B1 = np.zeros([n_x, n_w])
        self._B2 = np.zeros([n_x, n_u])

        # regularized output: z(t) = C1 * x(t) + D11 * w(t) + D12 * u(t)
        self._C1 = np.zeros([n_z, n_x])
        self._D11 = np.zeros([n_z, n_w])
        self._D12 = np.zeros([n_z, n_u])

        # measurement: y(t) = C2 * x(t) + D21 * w(t) + D22 * u(t)
        self._C2 = np.zeros([n_y, n_x])
        self._D21 = np.zeros([n_y, n_w])
        self._D22 = np.zeros([n_y, n_u])

    def initialize(self, x0=None, hat_w=None):
        SystemModel.initialize(self)

        if x0 is None:
            # use the previous x0
            if self._x0 is None:
                # use zero initialization if self._n_x is set
                if self._n_x > 0:
                    self._x0 = np.zeros([self._n_x, 1])
            x0 = self._x0
        else:
            self._x0 = x0

        # set x0
        self._n_x = x0.shape[0]
        self._x = x0

        # initializing output and measurements by treating w and u to be zeros.
        # one might change this part for some other initialization strategies
        if not self._ignore_output:
            if self._C1 is None:
                self.errorMessage('C1 is not defined when the system output (z) is not ignored. Initialization fails.')
            else:
                self._z = np.dot(self._C1, self._x)

        if not self._state_feedback:
            if self._C2 is None:
                self.errorMessage('C2 is not defined for an output-feedback system. Initialization fails.')
            else:
                self._y = np.dot(self._C2, self._x)

    def measurementConverge(self, u, w=None, hat_w=None):
        if w is not None:
            if w.shape[0] != self._n_w:
                return self.errorMessage('Dimension mismatch: w')

            if not isinstance(w, np.ndarray):
                w = np.array(w)
            if not isinstance(hat_w, np.ndarray):
                hat_w = np.array(hat_w)

            if not self._state_feedback:
                # TODO
                pass
        else:
            if hat_w is not None:
                return self.errorMessage('Invalid prediction: hat_w')

            if not self._state_feedback:
                # TODO
                pass

        return self._y

    def systemProgress(self, u, w=None):
        if w.shape[0] != self._n_w:
            return self.errorMessage('Dimension mismatch: w')
        self._x = (
                np.dot(self._A, self._x) +
                np.dot(self._B2, u) +
                np.dot(self._B1, w)
        )


    def updateAction(self, new_act_ids=None):
        if new_act_ids is None:
            new_act_ids = []
        sys = self.system_copy()
        sys._B2 = self._B2[:, new_act_ids]
        sys._D12 = self._D12[:, new_act_ids]
        sys._n_u = len(new_act_ids)
        return sys

    def system_copy(self):
        sys = LTI_System()

        # state
        sys._A = self._A
        sys._B1 = self._B1

        # reg output
        sys._C1 = self._C1
        sys._D11 = self._D11
        sys._D12 = self._D12

        # measurement
        sys._C2 = self._C2
        sys._D22 = self._D22

        # vector dimensions
        sys._n_x = self._n_x
        sys._n_z = self._n_z
        sys._n_y = self._n_y
        sys._n_w = self._n_w

        return sys




