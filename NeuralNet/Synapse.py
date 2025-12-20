import numpy as np
from NeuralNet.Neuron import Neuron

class Synapse:
    def __init__(self, weight, pre: Neuron, post: Neuron, lr = 0.01, aPlus = 10.0, aMinus = 10.0, maxW = 10.0, traceDecay = 0.95):
        self.weight = weight
        self.preSpike = pre
        self.postSpike = post
        self.lr = lr
        self.aPlus = aPlus
        self.aMinus = aMinus
        self.maxW = maxW
        self.traceDecay = traceDecay
        self.trace = 0.0  

    def applySTDP(self, currentTime):
        preSpikeTrace = self.preSpike.spikeTrace
        postSpikeTrace = self.postSpike.spikeTrace

        delta = 0.0
        if self.preSpike.spikeTime == currentTime:
            delta -= self.aMinus * postSpikeTrace
            self.trace -= self.aMinus * postSpikeTrace
        if self.postSpike.spikeTime == currentTime:
            delta += self.aPlus * preSpikeTrace
            self.trace += self.aPlus * preSpikeTrace

        self.trace *= self.traceDecay
        self.weight += self.lr * delta
        self.weight = np.clip(self.weight, -self.maxW, self.maxW)        
        return self.weight
        
    def handleReward(self, reward):
        delta = self.lr * self.trace * reward
        self.weight += delta
        self.weight = np.clip(self.weight, -self.maxW, self.maxW)
        return self.weight
    
    def getCurrent(self, spike):
        return self.weight * spike
