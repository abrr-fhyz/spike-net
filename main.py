import numpy as np
from NeuralNet.SNN import SNN
from Data.Connectome import buildNervousSystem, buildSynapticStructure

#Sample SNN Example
#from NeuralNet.SNNBeta import SNN
#steps = 100
#i = 6
#h = 7
#o = 4
#model = SNN(steps, i, h, o)
#model.train()

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