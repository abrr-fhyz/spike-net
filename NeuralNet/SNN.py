import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from NeuralNet.Neuron import Neuron
from NeuralNet.Synapse import Synapse

class SNN:
    def __init__(self, timeSteps, inputSize, hidden, outputSize, pattern):
        self.currentTime = 0
        self.timeSteps = timeSteps
        self.pattern = pattern

        self.inputSize = inputSize
        self.hidden = hidden
        self.outputSize = outputSize

        self._initialize()

    def _initialize(self):
        self.inputNeurons = [Neuron() for i in range(self.inputSize)]
        self.hiddenNeurons = [Neuron() for i in range(self.hidden)]
        self.outputNeurons = [Neuron() for i in range(self.outputSize)]

        self.inputToHiddenWeights = self._createWeights(self.inputSize, self.hidden).tocsr()
        self.hiddenToOutputWeights = self._createWeights(self.hidden, self.outputSize).tocsr()

        self.inputHiddenSynapses = {}
        self.hiddenOutputSynapses = {}

        ihWeights = self.inputToHiddenWeights.tolil()
        for idx in range(self.inputSize):
            for jdx in ihWeights.rows[idx]:
                weight = ihWeights[idx, jdx]
                self.inputHiddenSynapses[(idx, jdx)] = Synapse(weight, self.inputNeurons[idx], self.hiddenNeurons[jdx])
        hoWeights = self.hiddenToOutputWeights.tolil()
        for idx in range(self.hidden):
            for jdx in hoWeights.rows[idx]:
                weight = hoWeights[idx, jdx]
                self.hiddenOutputSynapses[(idx, jdx)] = Synapse(weight, self.hiddenNeurons[idx], self.outputNeurons[jdx])

    def _createWeights(self, rows, cols):
        #randomly generated for now
        weights = lil_matrix((rows, cols))
        for idx in range(rows):
            nConnections = max(1, int(cols * 0.4))
            connected = np.random.choice(cols, size=nConnections, replace=False)
            for jdx in connected:
                weights[idx, jdx] = np.random.rand() * 2.0 + 0.3
        return weights

    def vectorisedUpdate(self, neurons, inputSpikes, weights):
        neuronsLen = len(neurons)
        outputSpikes = np.zeros(neuronsLen, dtype=bool)
        weightedInput = weights.T.dot(inputSpikes)
        for idx in range(len(neurons)):
            neuron = neurons[idx]
            outputSpikes[idx] = neuron.update(weightedInput[idx], self.currentTime)
        return outputSpikes.astype(float)

    def forward(self, inputSpikes):
        hiddenSpikes = self.vectorisedUpdate(self.hiddenNeurons, inputSpikes, self.inputToHiddenWeights)
        outputSpikes = self.vectorisedUpdate(self.outputNeurons, hiddenSpikes, self.hiddenToOutputWeights)
        return outputSpikes

    def stdp(self, synapsesDict: dict[tuple[int, int], Synapse], weightMatrix):
        weightMatrix = weightMatrix.tolil()
        for (idx, jdx) in synapsesDict:
            synapse = synapsesDict[(idx, jdx)]
            newWeight = synapse.applySTDP(self.currentTime)
            weightMatrix[idx, jdx] = newWeight
        return weightMatrix.tocsr() 

    def calculateReward(self, output, targetRate = 0.3):
        reward = 0

        firingRate = np.sum(output) / len(output)
        reward += 1.0 - abs(firingRate - targetRate) / max(targetRate, 0.1)

        avgTrace = np.mean([neuron.spikeTrace for neuron in self.outputNeurons])
        reward += 1.0 - abs(avgTrace - targetRate)

        reward /= 2
        return reward 

    def processReward(self, reward, synapsesDict: dict[tuple[int, int], Synapse], weightMatrix):
        weightMatrix = weightMatrix.tolil()
        for (idx, jdx) in synapsesDict:
            synapse = synapsesDict[(idx, jdx)]
            newWeight = synapse.handleReward(reward)
            weightMatrix[idx, jdx] = newWeight
        return weightMatrix.tocsr()
    
    def train(self):
        for step in range(self.timeSteps):
            self.currentTime = step
            inputSpikes = np.random.randint(0, 2, size=self.inputSize).astype(float)
            outputSpikes = self.forward(inputSpikes)
            self.inputToHiddenWeights = self.stdp(self.inputHiddenSynapses, self.inputToHiddenWeights)
            self.hiddenToOutputWeights = self.stdp(self.hiddenOutputSynapses, self.hiddenToOutputWeights)
            reward = self.calculateReward(outputSpikes)
            self.inputToHiddenWeights = self.processReward(reward, self.inputHiddenSynapses, self.inputToHiddenWeights)
            self.hiddenToOutputWeights = self.processReward(reward, self.hiddenOutputSynapses, self.hiddenToOutputWeights)
            print(f"TimeStep: {step} | Output: {outputSpikes} | Reward: {reward:.3f}")
            