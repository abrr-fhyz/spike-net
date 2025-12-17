import numpy as np

class Neuron:
    def __init__(self, threshold = 1.0, reset = 0.0, decay = 0.95, refractory = 2, leakConductance = 0.1, restPotential = 0.0):
        self.threshold = threshold
        self.reset = reset
        self.decay = decay
        self.refractory = refractory
        self.membranePotential = 0
        self.spikeTime = -1
        self.refractoryEnd = -1

        self.leakConductance = leakConductance
        self.restPotential = restPotential

        self.spikeTrace = 0.0

    def setName(self, name):
        self.neuronName = name

    def triggerSpike(self, time):
        self.spikeTime = time
        self.membranePotential = self.reset
        self.refractoryEnd = time + self.refractory
        self.spikeTrace += 1.0

    def update(self, incomingSpikes, currentTime):
        if currentTime < self.refractoryEnd:
            return False
        
        self.membranePotential += incomingSpikes
        leakCurrent = self.leakConductance * (self.membranePotential - self.restPotential)
        self.membranePotential -= leakCurrent
        self.spikeTrace *= self.decay

        if self.membranePotential >= self.threshold:
            self.triggerSpike(currentTime)
            return True
        
        return False
    