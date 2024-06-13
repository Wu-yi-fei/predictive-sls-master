from base_process import NoiseModel
import numpy as np
'''
To create a new noise model, inherit the following base function and customize the specified methods.

class NoiseModel:
    def __init__ (self, n_w=0):
    def initialize (self):
    def getNoise (self,**kwargs):
        # the noise can depend on some parameters such as state or control
        return w
'''

class ZeroNoise (NoiseModel):
    '''
    Generate zero vector as the noise
    '''
    def __init__ (self, n_w=0):
        self.setDimension(n_w)
    
    def setDimension(self, n_w=0):
        self._n_w = n_w
        self._w = np.zeros([n_w,1])

    def getNoise(self,**kwargs):
        return self._w.copy()


class GaussianNoise(NoiseModel):
    '''
    Generate Gaussian noise
    '''
    def __init__ (self, n_w=0, mu=0, sigma=1):
        NoiseModel.__init__(self,n_w=n_w)

        self._mu = mu
        self._sigma = sigma
    
    def getNoise (self,**kwargs):
        return np.random.normal (self._mu, self._sigma, (self._n_w,1))
    
    
class FixedNoiseVector(NoiseModel):
    '''
    Fixed noise vector
    '''
    def __init__ (self, n_w=0, horizon=0, t0=0, FIR_horizon=0):
        NoiseModel.__init__(self, n_w=n_w)
        self._horizon = horizon
        self._fir_horizon = FIR_horizon
        self._t = 0
        self._t0 = 0
        self._t1 = 0
        self._w = []

    def initialize (self):
        self.startAtTime(t=self._t0)

    def startAtTime(self, t=0):
        self._t = self._t0 = t
        self._t1 = self._t0 = t

    def generateNoiseFromNoiseModelInstance(self, noise_model=None):
        if not isinstance (noise_model, NoiseModel):
            return

        self._n_w = noise_model._n_w

        self._w = []
        for t in range (self._horizon):
            self._w.append(noise_model.getNoise())
        
    def generateNoiseFromNoiseModel (self, cls=NoiseModel):
        noise_model = cls(n_w=self._n_w)
        self.generateNoiseFromNoiseModelInstance(noise_model=noise_model)
    
    def setNoise(self,w=None):
        # directly assign the _w vector
        self._w = w

    def getNoise(self,**kwargs):
        if self._t < self._horizon:
            w = self._w[self._t]
            self._t += 1
            return w

        return np.zeros((self._n_w,1))

    def getFutureNoise(self, **kwargs):
        if self._t1 < self._horizon - self._fir_horizon:
            w = self._w[self._t1 + self._fir_horizon]
            self._t1 += 1
            return w
        return np.zeros((self._n_w, 1))

    def getNoiseTrajectroy(self, **kwargs):
        ws = self._w[:self._fir_horizon]
        return ws


class MixedNoise(NoiseModel):
    '''
    Combine different noise models
    '''
    def __init__(self, n_w, *argv):
        NoiseModel.__init__(self, n_w=n_w)
        self._noise_models = argv
    
    def initialize(self):
        for noise_model in self._noise_models:
            noise_model.initialize()


class MixedNoiseConcat(MixedNoise):
    '''
    Concatenate different noise models together in the order they are inputted in
    '''
    def __init__(self, *argv):
        n_w = 0
        for arg in argv:
            n_w += arg._n_w
        MixedNoise.__init__(self, n_w, *argv)

    def getNoise(self, **kwargs):
        totalNoise = []
        for noise_model in self._noise_models:
            totalNoise.append(noise_model.getNoise(**kwargs))
        return np.concatenate(totalNoise)


class MixedNoiseAdd(MixedNoise):
    '''
    Add noise models together. Models must have same n_w
    '''
    def __init__(self, *argv):
        n_w = argv[0]._n_w
        MixedNoise.__init__(self, n_w, *argv)
            
    def getNoise(self, **kwargs):
        totalNoise = np.zeros((self._n_w, 1))
        for noise_model in self._noise_models:
            totalNoise += noise_model.getNoise(**kwargs)
        return totalNoise