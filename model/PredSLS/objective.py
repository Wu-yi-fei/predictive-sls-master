from xml.sax import SAXParseException

from model.base_optimize import *
import control as ctrl
import numpy as np
import cvxpy as cp


class SLS_Obj_LQ(SLS_Objective):
    '''
    return

        || [ [Q]^0.5, [R]^0.5 ][ Phi_x; Phi_u ] ||_F^2
        = \sum_{k=0}^T (|| Q [ Phi_x[k] ] ||_F^2 + || R [ Phi_u[k] ] ||_F^2)
        i.e. LQR for state-feedback SLS

        || [ [Q]^0.5, [R]^0.5 ][ Phi_x + Phi_hat_x; Phi_u + Phi_hat_u ] ||_F^2

                 [[ Phi_hat_x[0,0]+Phi_x[0] Phi_hat_x[0,1]             ...  Phi_hat_x[0,T]          ] * [ Q             ] [[ Phi_hat_x[0,0]+Phi_x[0] Phi_hat_x[0,1]             ...  Phi_hat_x[0,T]          ]
        = trace(  [ Phi_hat_x[1,0]+Phi_x[1] Phi_hat_x[1,1]+Phi_x[0]    ...  Phi_hat_x[1,T]          ]   [    ...        ]  [ Phi_hat_x[1,0]+Phi_x[1] Phi_hat_x[1,1]+Phi_x[0]    ...  Phi_hat_x[1,T]          ]
                  [              ...                ...                         ...                 ]   [         Q     ]  [              ...                ...                         ...                 ]
                  [ Phi_hat_x[T,0]+Phi_x[T] Phi_hat_x[T,1]+Phi_x[T-1]  ...  Phi_hat_x[T,T]+Phi_x[0] ]]  [            ...]  [ Phi_hat_x[T,0]+Phi_x[T] Phi_hat_x[T,0]+Phi_x[T-1]  ...  Phi_hat_x[T,T]+Phi_x[0] ]]

                 +  [[ Phi_hat_u[0,0]+Phi_u[0] Phi_hat_u[0,1]             ...  Phi_hat_u[0,T]          ] * [ R             ] [[ Phi_hat_u[0,0]+Phi_u[0] Phi_hat_u[0,1]             ...  Phi_hat_u[0,T]          ]
        =            [ Phi_hat_u[1,0]+Phi_u[1] Phi_hat_u[1,1]+Phi_u[0]    ...  Phi_hat_u[1,T]          ]   [    ...        ]  [ Phi_hat_u[1,0]+Phi_u[1] Phi_hat_u[1,1]+Phi_x[0]    ...  Phi_hat_u[1,T]          ]
                     [              ...                ...                         ...                 ]   [         R     ]  [              ...                ...                         ...                 ]
                     [ Phi_hat_u[T,0]+Phi_u[T] Phi_hat_u[T,1]+Phi_u[T-1]  ...  Phi_hat_u[T,T]+Phi_u[0] ]]  [            ...]  [ Phi_hat_u[T,0]+Phi_u[T] Phi_hat_u[T,0]+Phi_x[T-1]  ...  Phi_hat_u[T,T]+Phi_x[0] ]]
                 )
        = \sum_{t=0}^T \sum_{k=0}^t (|| Q [ Phi_hat_x[t,k]+Phi_x[k] ] ||_F^2 + || R [ Phi_hat_u[t,k]+Phi_u[k] ] ||_F^2)
          + \sum_{t=0}^T \sum_{k=t+1}^T (|| Q [ Phi_hat_x[t,k] ] ||_F^2 + || R [ Phi_hat_u[t,k] ] ||_F^2)


        i.e. LQR for predictive state-feedback SLS
    '''

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
        if sls._predictive:
            # predictive feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u
            Phi_hat_x = sls._Phi_hat_x
            Phi_hat_u = sls._Phi_hat_u
            for t in range(1, sls._fir_horizon + 1):
                for k in range(1, sls._fir_horizon + 1):
                    self._objective_expression += cp.sum_squares(
                            Q_sqrt @ Phi_x[t * (sls._fir_horizon + 1) + k] +
                            Q_sqrt @ Phi_hat_x[(t-1) * sls._fir_horizon + k - 1]) + cp.sum_squares(
                            R_sqrt @ Phi_u[t * (sls._fir_horizon + 1) + k] +
                            R_sqrt @ Phi_hat_u[(t-1) * sls._fir_horizon + k - 1]
                    )
        else:
            # feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u

            for t in range(1, sls._fir_horizon + 1):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ Phi_x[sls._fir_horizon * (sls._fir_horizon + 1) + t]) + cp.sum_squares(
                    R_sqrt @ Phi_u[sls._fir_horizon * (sls._fir_horizon + 1) + t]
                )
                # for k in range(1, sls._fir_horizon + 1):
                #     self._objective_expression += cp.sum_squares(
                #         Q_sqrt @ Phi_x[t * (sls._fir_horizon + 1) + k]) + cp.sum_squares(
                #         R_sqrt @ Phi_u[t * (sls._fir_horizon + 1) + k]
                #     )
        return objective_value + self._objective_expression


class SLS_Obj_LQ_Dist(SLS_Objective):

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
        if sls._predictive:
            phi_x = sls._phi_x
            phi_u = sls._phi_u
            phi_hat_x = sls._phi_hat_x
            phi_hat_u = sls._phi_hat_u
            for t in range(1, sls._fir_horizon + 1):
                for k in range(1, sls._fir_horizon + 1):
                    self._objective_expression += cp.sum_squares(
                        Q_sqrt @ (phi_x[t * (sls._fir_horizon + 1) + k] + phi_hat_x[(t - 1) * sls._fir_horizon + k - 1])
                    ) + cp.sum_squares(
                        R_sqrt @ (phi_u[t * (sls._fir_horizon + 1) + k] + phi_hat_u[(t - 1) * sls._fir_horizon + k - 1])
                    )
        else:
            phi_x = sls._phi_x
            phi_u = sls._phi_u

            for t in range(1, sls._fir_horizon + 1):
                self._objective_expression += cp.sum_squares(
                    Q_sqrt @ phi_x[sls._fir_horizon * (sls._fir_horizon + 1) + t]) + cp.sum_squares(
                    R_sqrt @ phi_u[sls._fir_horizon * (sls._fir_horizon + 1) + t]
                )
        return objective_value + self._objective_expression

class LQC(SLS_Objective):

    def __init__(self, A, B, Q=None, R=None):
        super().__init__()
        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

    def ObjectiveValue(self, alg=None, us=None, xs=None):
        A = self._A
        B = self._B
        Q = self._Q
        R = self._R
        # default values
        if Q is None:
            Q = alg._system_model._C1
        if R is None:
            R = alg._system_model._D12

        self._objective_expression = 0
        # objective function
        # print(len(xs))
        # print(len(us))
        for t in range(len(xs)):
            if t == len(xs) - 1:
                _P, _, _ = ctrl.dare(A, B, Q, R)
                self._objective_expression += (np.matmul(np.transpose(R @ us[t]), us[t])
                                               + np.matmul(np.transpose( Q @ xs[t]), xs[t]))
            else:
                self._objective_expression += (np.matmul(np.transpose(R @ us[t]), us[t])
                                               + np.matmul(np.transpose( Q @ xs[t]), xs[t]))
            # print(self._objective_expression)
            # self._objective_expression += np.square(np.linalg.norm(                                            + np.matmul(np.transpose(Q_sqrt @ xs[t]), (Q_sqrt @ xs[t]))
            #     Q_sqrt @ xs[t]
            # )) + np.square(np.linalg.norm(
            #     R_sqrt @ us[t]
            # ))
        return self._objective_expression

class SLS_Obj_LQ_w(SLS_Objective):
    '''
    return
        || [ [Q]^0.5, [R]^0.5 ][ Phi_x + Phi_hat_x; Phi_u + Phi_hat_u ] w ||_F^2
    '''

    def __init__(self, Q_sqrt=None, R_sqrt=None):
        super().__init__()
        self._Q_sqrt = Q_sqrt
        self._R_sqrt = R_sqrt

    def ObjectiveValue(self, alg, ws, ws_hat, Xs, pre):
        Q = self._Q_sqrt
        R = self._R_sqrt
        if ws_hat is None:
            ws_hat = ws
        self._objective_expression = 0
        # predictive feedback
        Phi_x = alg._controller._Phi_x
        Phi_u = alg._controller._Phi_u
        if pre:
            Phi_hat_x = alg._controller._Phi_hat_x
            Phi_hat_u = alg._controller._Phi_hat_u
        else:
            Phi_hat_x = 0
            Phi_hat_u = 0
        # print(Phi_hat_x)
        # print(np.shape(Phi_hat_x))
        for t in range(len(Phi_x)):
            xs = 0
            us = 0
            for k in range(len(Phi_x)):
                if pre:
                    if k <= t:
                        xs += Phi_x[k] @ ws[t - k]
                        us += Phi_u[k] @ ws[t - k]
                        xs += Phi_hat_x[t * len(Phi_x) + t - k + 1] @ ws_hat[t - k]
                        us += Phi_hat_u[t * len(Phi_u) + t - k + 1] @ ws_hat[t - k]
                    else:
                        xs += Phi_hat_x[t * len(Phi_x) + k] @ ws_hat[k - 1]
                        us += Phi_hat_u[t * len(Phi_u) + k] @ ws_hat[k - 1]
                else:
                    if k <= t:
                        xs += Phi_x[k] @ ws[t - k]
                        us += Phi_u[k] @ ws[t - k]

                # self.xs += np.matmul(Phi_hat_x[t * len(Phi_x) + k], ws_hat[k - 1])
                # self.us += np.matmul(Phi_hat_u[t * len(Phi_u) + k], ws_hat[k - 1])
            # print("xs", xs)
            xs = Xs[t]
            self._objective_expression += np.transpose(xs) @ Q @ xs + np.transpose(us) @ R @ us
                    # self._objective_expression += np.matmul(np.matmul(np.transpose(np.matmul(Phi_hat_x[t * len(Phi_x) + k],ws[k - 1])), Q), np.matmul(Phi_hat_x[t * len(Phi_x) + k], ws[k-1]))\
                    #                               + np.matmul(np.matmul(np.transpose(np.matmul(Phi_hat_u[t * len(Phi_x)  + k], ws[k - 1])), R), np.matmul(Phi_hat_u[t * len(Phi_x) + k], ws[k-1]))

        return self._objective_expression