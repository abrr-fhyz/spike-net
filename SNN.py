import numpy as np
from Neuron import Neuron, Synapse

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
        # need to streamline this to be less rigid
        self.inputNeurons = [Neuron() for i in range(self.inputSize)]
        self.hiddenNeurons = [Neuron() for i in range(self.hidden)]
        self.outputNeurons = [Neuron() for i in range(self.outputSize)]
        self.input_to_hidden_synapses = np.random.rand(self.inputSize, self.hidden)
        self.hidden_to_output_synapses = np.random.rand(self.hidden, self.outputSize)

    def stdp(self, preSpike, postSpike, weight, lr = 0.01, tau_plus = 20, tau_minus = 20):
        if preSpike > 0 and postSpike > 0:
            delta = postSpike - preSpike
            if delta > 0:
                return weight + lr * np.exp(-delta / tau_plus)
            else:
                return weight - lr * np.exp(delta / tau_minus)
        return weight
        
    def train(self):
        for step in range(self.timeSteps):
            inputSpikes = np.random.randint(0, 2, size = self.inputSize) # needs to take in input
            hiddenSpikes = np.zeros(self.hidden)
            for idx in range(len(self.inputNeurons)):
                neuron = self.inputNeurons[idx]
                if neuron.update(inputSpikes[idx] * self.input_to_hidden_synapses[idx], step):
                    hiddenSpikes += self.input_to_hidden_synapses[idx]
            outputSpikes = np.zeros(self.outputSize)
            for idx in range(len(self.hiddenNeurons)):
                neuron = self.hiddenNeurons[idx]
                if neuron.update(hiddenSpikes[idx] * self.hidden_to_output_synapses[idx], step):
                    outputSpikes += self.hidden_to_output_synapses[idx]
            for idx in range(len(self.outputNeurons)):
                neuron = self.outputNeurons[idx]
                neuron.update(outputSpikes[idx], step)

            for idx in range(self.inputSize):
                for jdx in range(self.hidden):
                    self.input_to_hidden_synapses[idx, jdx] = self.stdp(self.inputNeurons[idx].spikeTime, self.hiddenNeurons[jdx].spikeTime, self.input_to_hidden_synapses[idx, jdx])
            for idx in range(self.hidden):
                for jdx in range(self.outputSize):
                    self.hidden_to_output_synapses[idx, jdx] = self.stdp(self.hiddenNeurons[idx].spikeTime, self.outputNeurons[jdx].spikeTime, self.hidden_to_output_synapses[idx, jdx])
            
            # need to implement reward function here      
            if all(neuron.spikeTime == step for neuron, pat in zip(self.inputNeurons, self.pattern) if pat == 1):
                print(f"Pattern detected at timestep {step}.")

            