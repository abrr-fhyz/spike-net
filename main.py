import numpy as np
from NeuralNet.SNNBeta import SNN
from Data.Connectome import buildNervousSystem, buildSynapticStructure

synapses, sensoryNeurons, interNeurons, motorNeurons = buildNervousSystem()
sensoryToInterSynapses, interToMotorSynapses, ssSynapses, iiSynapses, mmSynapses, motorToInterSynapses, interToSensorySynapses = buildSynapticStructure(synapses)

print("========== GENERATING SYSTEM INPUT =========")
steps = 100
spikeTrain = []
for i in range(steps):
    spike = np.random.randint(0, 2, size=len(sensoryNeurons)).astype(float)
    spikeTrain.append(spike)

print("========== BOOTING UP NERVOUS SYSTEM =========")
nervousSystem = SNN(timeSteps=steps)
nervousSystem.initialize(sensoryNeurons, interNeurons, motorNeurons, sensoryToInterSynapses, interToMotorSynapses, ssSynapses, iiSynapses, mmSynapses, motorToInterSynapses, interToSensorySynapses)
nervousSystem.train(spikeTrain)