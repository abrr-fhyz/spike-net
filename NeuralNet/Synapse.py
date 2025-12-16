import numpy as np
from NeuralNet.Neuron import Neuron

class Synapse:
    def __init__(self, weight, pre: Neuron, post: Neuron, lr = 0.01, tauPlus = 20.0, tauMinus = 20.0):
        self.weight = weight
        self.preSpike = pre
        self.postSpike = post
        self.lr = lr
        self.tauPlus = tauPlus
        self.tauMinus = tauMinus

    def applySTDP(self):
        preSpikeTime = self.preSpike.spikeTime
        postSpikeTime = self.postSpike.spikeTime

        if preSpikeTime > 0 and postSpikeTime > 0:
            delta = preSpikeTime - postSpikeTime
            if delta > 0:
                self.weight += self.lr * np.exp(-delta / self.tauPlus)
            else:
                self.weight -= self.lr * np.exp(delta / self.tauMinus)
        
        return self.weight
    
    def getCurrent(self, spike):
        return self.weight * spike
