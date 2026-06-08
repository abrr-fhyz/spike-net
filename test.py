import numpy as np
from Env.chemotax import ChemotaxisEnvironment
from NeuralNet.SNN import SNN
from Data.Connectome import buildNervousSystem, buildSynapticStructure

synapses, sensoryNeurons, interNeurons, motorNeurons = buildNervousSystem()
sensoryToInterSynapses, interToMotorSynapses, ssSynapses, iiSynapses, mmSynapses, motorToInterSynapses, interToSensorySynapses = buildSynapticStructure(synapses)
env = ChemotaxisEnvironment(sensoryNeurons,motorNeurons)

print("========== BOOTING UP NERVOUS SYSTEM =========")
nervousSystem = SNN(timeSteps=10000)
nervousSystem.initialize(sensoryNeurons, interNeurons, motorNeurons, sensoryToInterSynapses, interToMotorSynapses, ssSynapses, iiSynapses, mmSynapses, motorToInterSynapses, interToSensorySynapses)

print("========== LETTING WORM FIND FOOD =========")
observation = env.reset()
for step in range(10000):
    nervousSystem.currentTime = step
    nervousSystem._backward()
    output = nervousSystem._forward(observation)
    nervousSystem._applySTDP()
    observation, reward, done = env.step(output)
    nervousSystem._applyReward(reward)
    distance = np.linalg.norm(env.position - env.food_position)
    print(f"Step={step:<6} Reward={reward:<10.6f} Distance={distance:<6.2f}", end='\r', flush=True)
    if abs(reward) < 0.0000001:
        print()
        print("Food Found")
        break

    if done:
        print()
        print(f"Food reached at step {step}")
        break