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

print("========== BOOTING UP NERVOUS SYSTEM =========")
nervousSystem = SNN(timeSteps=100)
nervousSystem.initialize(sensoryNeurons, interNeurons, motorNeurons, sensoryToInterSynapses, interToMotorSynapses, ssSynapses, iiSynapses, mmSynapses, motorToInterSynapses, interToSensorySynapses)
nervousSystem.train()