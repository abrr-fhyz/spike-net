import numpy as np

class Neuron:
    def __init__(self, threshold = 1.0, reset = 0.0, decay = 0.9, refractory = 2):
        self.threshold = threshold
        self.reset = reset
        self.decay = decay
        self.refractory = refractory
        self.membranePotential = 0
        self.spikeTime = -1
        self.refractoryEnd = -1

    def setName(self, name):
        self.neuronName = name

    def update(self, incomingSpikes, currentTime):
        if currentTime < self.refractoryEnd:
            return False
        
        self.membranePotential *= self.decay
        self.membranePotential += np.sum(incomingSpikes)

        if self.membranePotential >= self.threshold:
            self.spikeTime = currentTime
            self.membranePotential = self.reset
            self.refractoryEnd = currentTime + self.refractory
            return True
        
        return False
    
class Synapse:
    def __init__(self, weight):
        self.weight = weight
