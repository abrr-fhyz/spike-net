import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from NeuralNet.Neuron import Neuron
from NeuralNet.Synapse import Synapse

class SNN:
    def __init__(self, timeSteps, inputSize, hidden, outputSize):
        self.currentTime = 0
        self.timeSteps = timeSteps

        self.inputSize = inputSize
        self.hidden = hidden
        self.outputSize = outputSize

        self.initializeRandom()

    def initializeRandom(self):
        self.inputNeurons = [Neuron() for i in range(self.inputSize)]
        self.hiddenNeurons = [Neuron() for i in range(self.hidden)]
        self.outputNeurons = [Neuron() for i in range(self.outputSize)]

        self.inputToHiddenWeights = self._createWeights(self.inputSize, self.hidden).tocsr()
        self.hiddenToOutputWeights = self._createWeights(self.hidden, self.outputSize).tocsr()
        self.interInputWeights = self._createWeights(self.inputSize, self.inputSize).tocsr()
        self.interHiddenWeights = self._createWeights(self.hidden, self.hidden).tocsr()
        self.interOutputWeights = self._createWeights(self.outputSize, self.outputSize).tocsr()

        self.inputHiddenSynapses = {}
        self.hiddenOutputSynapses = {}
        self.interInputSynapses = {}
        self.interHiddenSynapses = {}
        self.interOutputSynapses = {}

        self._createSynapses(self.inputToHiddenWeights, self.inputSize, self.inputHiddenSynapses, self.inputNeurons, self.hiddenNeurons)
        self._createSynapses(self.hiddenToOutputWeights, self.hidden, self.hiddenOutputSynapses, self.hiddenNeurons, self.outputNeurons)
        self._createSynapses(self.interInputWeights, self.inputSize, self.interInputSynapses, self.inputNeurons, self.inputNeurons)
        self._createSynapses(self.interHiddenWeights, self.hidden, self.interHiddenSynapses, self.hiddenNeurons, self.hiddenNeurons)
        self._createSynapses(self.interOutputWeights, self.outputSize, self.interOutputSynapses, self.outputNeurons, self.outputNeurons)

        self.prevInputSpikes = np.zeros(self.inputSize)
        self.prevHiddenSpikes = np.zeros(self.hidden)
        self.prevOutputSpikes = np.zeros(self.outputSize)  

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
        
        return outputSpikes

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

    def _applyReward(self, reward):
        self.inputToHiddenWeights = self.processReward(reward, self.inputHiddenSynapses, self.inputToHiddenWeights)
        self.hiddenToOutputWeights = self.processReward(reward, self.hiddenOutputSynapses, self.hiddenToOutputWeights)
        self.interInputWeights = self.processReward(reward, self.interInputSynapses, self.interInputWeights)
        self.interHiddenWeights = self.processReward(reward, self.interHiddenSynapses, self.interHiddenWeights)
        self.interOutputWeights = self.processReward(reward, self.interOutputSynapses, self.interOutputWeights)

    def _createSynapses(self, weightMatrix, size, syanpseDict, firstNeuron, secondNeuron):
        tempWeights = weightMatrix.tolil()
        for idx in range(size):
            for jdx in tempWeights.rows[idx]:
                weight = tempWeights[idx, jdx]
                syanpseDict[(idx, jdx)] = Synapse(weight, firstNeuron[idx], secondNeuron[jdx])

    def _createWeights(self, rows, cols):
        #randomly generated for now
        weights = lil_matrix((rows, cols))
        for idx in range(rows):
            nConnections = max(1, int(cols * 0.4))
            connected = np.random.choice(cols, size=nConnections, replace=False)
            for jdx in connected:
                weights[idx, jdx] = np.random.rand() * 2.0 + 0.3
        return weights

    def train(self):
        for step in range(self.timeSteps):
            self.currentTime = step
            inputSpikes = np.random.randint(0, 2, size=self.inputSize).astype(float)
            outputSpikes = self._forward(inputSpikes)
            self._applySTDP()
            reward = self._calculateReward(outputSpikes)
            self._applyReward(reward)
            print(f"TimeStep: {step} | Output: {outputSpikes} | Reward: {reward:.3f}")
            