import numpy as np
import inspect

'''
The core abstract classes that form the SLSpy framework:

    SystemModel -> SynthesisAlgorithm -> ControllerModel

    SystemModel, ControllerModel, NoiseModel, PredictionModel -> Simulator -> Results
'''


class ObjBase:
    """
    The object base that defines debugging tools
    """
    def initialize(self, **kwargs):
        pass

    def sanityCheck(self):
        # check the system parameters are coherent
        return True

    def errorMessage(self, msg):
        print(self.__class__.__name__ + '-' + inspect.stack()[1][3] + ': [ERROR] ' + msg + '\n')
        return False

    def warningMessage(self, msg):
        print(self.__class__.__name__ + '-' + inspect.stack()[1][3] + ': [WARNING] ' + msg + '\n')
        return False

class SystemModel(ObjBase):
    '''
    The class for discrete-time system models.
    A controller synthesizer takes a system model and synthesizes a controller.
    '''

    def __init__(self,
                 predictable=False,
                 state_feedback=True,
                 ignore_output=False
                 ):
        self._x = np.empty([0])  # state
        self._y = np.empty([0])  # measurement
        self._z = np.empty([0])  # regularized output

        self.predictive_signal = predictable
        self._state_feedback = state_feedback
        self._ignore_output = ignore_output

        self._x0 = None

    def measurementConverge(self, u, w=None):
        # In an state-feedback system, the current control depends on the current state and predictions
        # e.g., u(t) = K * x(t) + L * hat_w(t)
        # while the current state is controlled by the controller from the previous step
        # e.g., x(t) = A * x(t-1) + B * u(t-1) + w(t-1)

        # Therefore, we need a convergence phase for them to agree with each other
        # This function allows the measurement to converge with the control
        # This function does not change the system's internal state
        return self._y

    def systemProgress(self, u, w=None, **kwargs):
        # This function takes the input (and the noise) and progress to next time
        # It also serves the purpose of convergence commitment,
        # i.e., it does change the system's internal state
        pass

    def getState(self):
        return self._x.copy()

    def getMeasurement(self):
        if self._state_feedback:
            return self._x.copy()
        else:
            return self._y.copy()

    def stateFeedback(self, state_feedback=True):
        # Is this model state-feedback?
        self._state_feedback = state_feedback

    def getOutput(self):
        if self._ignore_output:
            return None
        else:
            return self._z.copy()

    def ignoreOutput(self, ignore_output=False):
        # Does this model ignore output?
        self._ignore_output = ignore_output


class ControllerModel(ObjBase):
    '''
    The base class for discrete-time controller.
    '''
    def initialize(self, **kwargs):
        # Initialize internal
        pass

    def controlConvergence(self, x, **kwargs):
        # u(t) = K * x(t) + L * hat_w(t)
        return None

    def getControl(self, y, **kwargs):
        # This function commits the control.
        return None


class NoiseModel(ObjBase):
    '''
    The base class for noise model.
    NoiseModel is responsible for the right format of noise (dimension, etc.)
    '''

    def __init__(self, n_w=0):
        self._n_w = n_w  # dimension of the noise (disturbance)

    def initialize(self):
        # auto-initialization for each simulation
        pass

    def getNoise(self, **kwargs):
        # the noise can depend on some parameters such as state
        return 0


class PredictionModel(ObjBase):
    '''
    The base class for prediction model.
    NoiseModel is responsible for the right format of prediction (dimension, etc.)
    '''

    def __init__(self, n_w=0):
        self._n_w = n_w  # dimension of the prediction (disturbance)

    def initialize(self):
        # auto-initialization for each simulation
        pass

    def getPrediction(self, **kwargs):
        # the prediction can depend on some parameters such as controller
        return 0


class Algorithm(ObjBase):
    '''
    The base class for synthesis algorithm, which takes a system model and generates a controller model correspondingly.
    '''
    def __init__(self, system_model=None):
        self.setSystemModel(system_model=system_model)

    def __lshift__(self, system):
        return self.setSystemModel(system_model=system)

    def setSystemModel(self, system_model):
        if isinstance(system_model, SystemModel):
            self._system_model = system_model
        return self

    def synthesizedControllerModel(self):
        return None


class Simulator(ObjBase):
    '''
    Simulator
    '''
    def __init__(self,
                 system=None,
                 controller=None,
                 noise=None,
                 prediction=None,
                 horizon=-1,
                 convergence_threshold=1e-10
                 ):
        self.setSystem(system) # define system
        self.setController(controller) # define controller
        self.setNoise(noise) # define noise (Gaussian or Adversarial)
        self.setPrediction(prediction) # define prediction
        self.setHorizon(horizon) # define horizon
        self.setConvergenceThreshold(convergence_threshold) # define convergence condition
        pass

    def setSystem(self, system=None):
        if isinstance(system, SystemModel):
            self._system = system

    def setController(self, controller=None):
        if isinstance(controller, ControllerModel):
            self._controller = controller

    def setNoise(self, noise=None):
        if isinstance(noise, NoiseModel):
            self._noise = noise
        else:
            self._noise = None

    def setPrediction(self, prediction=None):
        if isinstance(prediction, NoiseModel):
            self._pred = prediction
        else:
            self._pred = None

    def setHorizon(self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def setConvergenceThreshold(self, convergence_threshold=1e-10):
        self._convergence_threshold = convergence_threshold

    def run(self, initialize=True):
        #   run the system and return
        #   system state (x)
        #   system measurement (y) # output feedback
        #   system output (z) # output feedback
        #   control history (u)
        #   noise history (w)
        #   provided prediction (hat_w)
        if self._horizon < 0:
            return None, None, None, None

        if not self._system.sanityCheck():
            return None, None, None, None

        if initialize:
            self._system.initialize()
            if self._noise is not None:
                self._noise.initialize()
            if self._pred is not None:
                self._pred.initialize()
                ws = self._pred.getNoiseTrajectroy()
                self._controller.initialize(hat_delta0=ws)
            else:
                self._controller.initialize()

        x_history = []
        y_history = []
        z_history = []
        u_history = []
        w_history = []

        need_convergence_phase = not self._system._state_feedback
        y = self._system.getMeasurement()

        for t in range(self._horizon):
            x = self._system.getState()
            if self._noise is not None:
                w = self._noise.getNoise()
                if self._pred is not None:
                    # hat_w = self._pred.getPrediction()
                    hat_w = self._pred.getFutureNoise()
                else:
                    hat_w = None
            else:
                w = None
                hat_w = None

            if need_convergence_phase:
                # TODO
                return self.errorMessage('Sorry, this experiment has not considered the output-feedback system')
            else:

                u = self._controller.getControl(y=y, hat_w=hat_w)

            self._system.systemProgress(u=u, w=w, hat_w=hat_w)

            y = self._system.getMeasurement()
            z = self._system.getOutput()

            x_history.append(x)
            y_history.append(y)
            z_history.append(z)
            u_history.append(u)
            w_history.append(w)
        return x_history, y_history, z_history, u_history, w_history, hat_w

class Causal_Offline_Optimal(ObjBase):
    '''
    Synthesizing the predictive controller using System Level Synthesis method.
    '''

    def __init__(self,  system=None,
                 controller=None,
                 noise=None,
                 horizon=0,
                 state_feedback=True,
                 predictive=True,
                 objective=None):
        self._state_feedback = state_feedback
        self._horizon = horizon
        self._predictive = predictive
        self._A = None
        self._B = None
        self._Q = None
        self._R = None
        self.setSystem(system)  # define system
        self.setController(controller)  # define controller
        self.setNoise(noise)  # define noise (Gaussian or Adversarial)
        self.setHorizon(horizon)  # define horizon


    def setSystem(self, system=None):
        if isinstance(system, SystemModel):
            self._system = system

    def setController(self, controller=None):
        if isinstance(controller, ControllerModel):
            self._controller = controller

    def setNoise(self, noise=None):
        if isinstance(noise, NoiseModel):
            self._noise = noise
        else:
            self._noise = None

    def setHorizon(self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def run(self, A, B, Q, R, initialize=True):

        self._use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        if initialize:
            self._system.initialize()
            if self._noise is not None:
                self._noise.initialize()
                self._controller.initialize()
            else:
                self._controller.initialize()
            if self._A == None:
                self._A = A
            if self._B == None:
                self._B = B
            if self._Q == None:
                self._Q = Q
            if self._R == None:
                self._R = R

        x_history = []
        u_history = []
        w_history = []

        y = self._system.getMeasurement()
        for t in range(self._horizon):
            x = self._system.getState()
            if self._noise is not None:
                w = self._noise.getNoise()
            else:
                w = None

            u = self._controller.getControl(y=y)

            self._system.systemProgress(u=u, w=w)

            y = self._system.getMeasurement()

            x_history.append(x)
            u_history.append(u)
            w_history.append(w)

        return x_history, u_history, w_history


class Noncausal_Offline_Optimal(ObjBase):
    '''
    Synthesizing the predictive controller using System Level Synthesis method.
    '''

    def __init__(self,  system=None,
                 controller=None,
                 noise=None,
                 prediction=None,
                 horizon=0,
                 state_feedback=True,
                 predictive=True,
                 objective=None):
        self._state_feedback = state_feedback
        self._horizon = horizon
        self._predictive = predictive
        self._A = None
        self._B = None
        self._Q = None
        self._R = None
        self.setSystem(system)  # define system
        self.setController(controller)  # define controller
        self.setNoise(noise)  # define noise (Gaussian or Adversarial)
        self.setPrediction(prediction)  # define prediction
        self.setHorizon(horizon)  # define horizon


    def setSystem(self, system=None):
        if isinstance(system, SystemModel):
            self._system = system

    def setController(self, controller=None):
        if isinstance(controller, ControllerModel):
            self._controller = controller

    def setNoise(self, noise=None):
        if isinstance(noise, NoiseModel):
            self._noise = noise
        else:
            self._noise = None

    def setPrediction(self, prediction=None):
        if isinstance(prediction, NoiseModel):
            self._pred = prediction
        else:
            self._pred = None

    def setHorizon(self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def run(self, A, B, Q, R, initialize=True):

        self._use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        if initialize:
            self._system.initialize()
            if self._noise is not None:
                self._noise.initialize()
            if self._pred is not None:
                self._pred.initialize()
                ws = self._pred.getNoiseTrajectroy()
                self._controller.initialize(hat_ws=ws)
            else:
                self._controller.initialize()
            if self._A == None:
                self._A = A
            if self._B == None:
                self._B = B
            if self._Q == None:
                self._Q = Q
            if self._R == None:
                self._R = R

        x_history = []
        u_history = []
        w_history = []

        y = self._system.getMeasurement()
        for t in range(self._horizon):
            x = self._system.getState()
            if self._noise is not None:
                w = self._noise.getNoise()
                if self._pred is not None:
                    # hat_w = self._pred.getPrediction()
                    hat_w = self._pred.getFutureNoise()
                else:
                    hat_w = None
            else:
                w = None
                hat_w = None

            u = self._controller.getControl(y=y, hat_w=hat_w)

            self._system.systemProgress(u=u, w=w, hat_w=hat_w)

            y = self._system.getMeasurement()

            x_history.append(x)
            u_history.append(u)
            w_history.append(w)

        return x_history, u_history, w_history, hat_w
