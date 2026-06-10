import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from NeuralNet.Neuron import Neuron
from NeuralNet.Synapse import Synapse

class SNN:
    def __init__(self, timeSteps, interval = 10):
        self.currentTime = 0
        self.timeSteps = timeSteps
        self.spikeInterval = interval
        self.forwardedOnce = False

    def initialize(self, inputNeurons, hiddenNeurons, outputNeurons, IHSynapses, HOSynapses, IISynapses, HHSynapses, OOSynapses, OHSynapses, HISynapses):
        self.inputNeurons = inputNeurons
        self.hiddenNeurons = hiddenNeurons
        self.outputNeurons = outputNeurons

        self.inputSize = len(inputNeurons)
        self.hidden = len(hiddenNeurons)
        self.outputSize = len(outputNeurons)

        self.inputHiddenSynapses = IHSynapses
        self.hiddenOutputSynapses = HOSynapses
        self.interInputSynapses = IISynapses
        self.interHiddenSynapses = HHSynapses
        self.interOutputSynapses = OOSynapses
        self.hiddenInputSynapses = HISynapses
        self.outputHiddenSynapses = OHSynapses

        self.prevInputSpikes = np.zeros(self.inputSize)
        self.prevHiddenSpikes = np.zeros(self.hidden)
        self.prevOutputSpikes = np.zeros(self.outputSize)

        self.inputToHiddenWeights = self._generateWeights(self.inputHiddenSynapses, self.inputSize, self.hidden).tocsr()
        self.hiddenToOutputWeights = self._generateWeights(self.hiddenOutputSynapses, self.hidden, self.outputSize).tocsr()
        self.interInputWeights = self._generateWeights(self.interInputSynapses, self.inputSize, self.inputSize).tocsr()
        self.interHiddenWeights = self._generateWeights(self.interHiddenSynapses, self.hidden, self.hidden).tocsr()
        self.interOutputWeights = self._generateWeights(self.interOutputSynapses, self.outputSize, self.outputSize).tocsr()
        self.hiddenInputWeights = self._generateWeights(self.hiddenInputSynapses, self.hidden, self.inputSize).tocsr()
        self.outputHiddenWeights = self._generateWeights(self.outputHiddenSynapses, self.outputSize, self.hidden).tocsr()

    def resetSNN(self):
        self.prevInputSpikes = np.zeros(self.inputSize)
        self.prevHiddenSpikes = np.zeros(self.hidden)
        self.prevOutputSpikes = np.zeros(self.outputSize)
        self.forwardedOnce = False
        self.currentTime = 0

    def stdp(self, synapsesDict: dict[tuple[int, int], Synapse], weightMatrix):
        weightMatrix = weightMatrix.tolil()
        for (idx, jdx) in synapsesDict:
            synapse = synapsesDict[(idx, jdx)]
            newWeight = synapse.applySTDP(self.currentTime)
            weightMatrix[idx, jdx] = newWeight
        return weightMatrix.tocsr() 

    def processReward(self, reward, synapsesDict: dict[tuple[int, int], Synapse], weightMatrix):
        weightMatrix = weightMatrix.tolil()
        for (idx, jdx) in synapsesDict:
            synapse = synapsesDict[(idx, jdx)]
            newWeight = synapse.handleReward(reward)
            weightMatrix[idx, jdx] = newWeight
        return weightMatrix.tocsr()

    def vectorisedUpdate(self, neurons, inputSpikes, weights, update = False, recurrentSpikes = None):
        neuronsLen = len(neurons)
        weightedInput = weights.T.dot(inputSpikes)
        if update:
            outputSpikes = np.zeros(neuronsLen, dtype=bool)
            if recurrentSpikes is not None:
                weightedInput += recurrentSpikes
            for idx in range(len(neurons)):
                neuron = neurons[idx]
                outputSpikes[idx] = neuron.update(weightedInput[idx], self.currentTime)
            return outputSpikes.astype(float)
        else:
            return weightedInput

    def _forward(self, inputSpikesTemp):
        inputSpikes = self.vectorisedUpdate(self.inputNeurons, self.prevInputSpikes, self.interInputWeights, update = True, recurrentSpikes = inputSpikesTemp)
        hiddenSpikesTemp = self.vectorisedUpdate(self.hiddenNeurons, inputSpikes, self.inputToHiddenWeights)
        hiddenSpikes = self.vectorisedUpdate(self.hiddenNeurons, self.prevHiddenSpikes, self.interHiddenWeights, update = True, recurrentSpikes = hiddenSpikesTemp)
        outputSpikesTemp = self.vectorisedUpdate(self.outputNeurons, hiddenSpikes, self.hiddenToOutputWeights)
        outputSpikes = self.vectorisedUpdate(self.outputNeurons, self.prevOutputSpikes, self.interOutputWeights, update = True, recurrentSpikes = outputSpikesTemp)

        self.prevInputSpikes = inputSpikes
        self.prevHiddenSpikes = hiddenSpikes
        self.prevOutputSpikes = outputSpikes
        self.forwardedOnce = True
        
        return outputSpikes
    
    def _backward(self):
        if not self.forwardedOnce:
            return
        self.prevHiddenSpikes = self.vectorisedUpdate(self.hiddenNeurons, self.prevOutputSpikes, self.outputHiddenWeights, update = True)
        self.prevInputSpikes = self.vectorisedUpdate(self.inputNeurons, self.prevHiddenSpikes, self.hiddenInputWeights, update = True)

    def _calculateReward(self, output, targetRate = 0.3):
        reward = 0

        firingRate = np.sum(output) / len(output)
        reward += 1.0 - abs(firingRate - targetRate) / max(targetRate, 0.1)

        avgTrace = np.mean([neuron.spikeTrace for neuron in self.outputNeurons])
        reward += 1.0 - abs(avgTrace - targetRate)

        reward /= 2
        return reward 
    
    def _applySTDP(self):
        self.inputToHiddenWeights = self.stdp(self.inputHiddenSynapses, self.inputToHiddenWeights)
        self.hiddenToOutputWeights = self.stdp(self.hiddenOutputSynapses, self.hiddenToOutputWeights)
        self.interInputWeights = self.stdp(self.interInputSynapses, self.interInputWeights)
        self.interHiddenWeights = self.stdp(self.interHiddenSynapses, self.interHiddenWeights)
        self.interOutputWeights = self.stdp(self.interOutputSynapses, self.interOutputWeights)
        self.hiddenInputWeights = self.stdp(self.hiddenInputSynapses, self.hiddenInputWeights)
        self.outputHiddenWeights = self.stdp(self.outputHiddenSynapses, self.outputHiddenWeights)

    def _applyReward(self, reward):
        self.inputToHiddenWeights = self.processReward(reward, self.inputHiddenSynapses, self.inputToHiddenWeights)
        self.hiddenToOutputWeights = self.processReward(reward, self.hiddenOutputSynapses, self.hiddenToOutputWeights)
        self.interInputWeights = self.processReward(reward, self.interInputSynapses, self.interInputWeights)
        self.interHiddenWeights = self.processReward(reward, self.interHiddenSynapses, self.interHiddenWeights)
        self.interOutputWeights = self.processReward(reward, self.interOutputSynapses, self.interOutputWeights)
        self.hiddenInputWeights = self.processReward(reward, self.hiddenInputSynapses, self.hiddenInputWeights)
        self.outputHiddenWeights = self.processReward(reward, self.outputHiddenSynapses, self.outputHiddenWeights)

    def _generateWeights(self, synapsesDict: dict[tuple[int, int], Synapse], rows, cols):
        weights = lil_matrix((rows, cols))
        for (idx, jdx) in synapsesDict:
            synapse = synapsesDict[(idx, jdx)]
            weight = synapse.weight
            weights[idx, jdx] = weight
        return weights
        
    def train(self, spikeTrain):
        for step in range(self.timeSteps):
            self.currentTime = step
            inputSpikes = spikeTrain[step]
            self._backward()
            outputSpikes = self._forward(inputSpikes)
            self._applySTDP()
            reward = self._calculateReward(outputSpikes)
            self._applyReward(reward)
            print(f"TimeStep: {step} | Output: {outputSpikes} | Reward: {reward:.3f}")
            