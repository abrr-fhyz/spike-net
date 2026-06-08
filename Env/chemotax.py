import numpy as np


class ChemotaxisEnvironment:

    def __init__(self, sensory_neurons, motor_neurons, world_size=200, food_position=(160, 160)):
        self.sensory_neurons = sensory_neurons
        self.motor_neurons = motor_neurons
        self.world_size = world_size
        self.food_position = np.array(food_position, dtype=float)

        self.buildNeuronMaps()
        self.reset()

    def buildNeuronMaps(self):
        self.sensoryMap = {
            n.neuronName: n.idx
            for n in self.sensory_neurons
        }
        self.motorMap = {
            n.neuronName: n.idx
            for n in self.motor_neurons
        }
        self.leftMotors = []
        self.rightMotors = []

        for neuron in self.motor_neurons:
            if neuron.neuronName.endswith("L"):
                self.leftMotors.append(neuron.idx)
            elif neuron.neuronName.endswith("R"):
                self.rightMotors.append(neuron.idx)

    def reset(self):
        self.position = np.array([40.0, 40.0])
        self.heading = np.random.uniform(0, 2*np.pi)
        self.previousConcentration = self.foodConcentration(self.position)
        return self.observe()

    def foodConcentration(self,point):
        distance = np.linalg.norm(point - self.food_position)
        return np.exp(-distance / 30.0)

    def observe(self):
        sensorDistance = 4.0
        leftSensor = self.position + np.array([np.cos(self.heading + 0.5), np.sin(self.heading + 0.5)]) * sensorDistance
        rightSensor = self.position + np.array([np.cos(self.heading - 0.5), np.sin(self.heading - 0.5)]) * sensorDistance
        centerSensor = self.position + np.array([np.cos(self.heading), np.sin(self.heading)]) * sensorDistance

        leftConc = self.foodConcentration(leftSensor)
        rightConc = self.foodConcentration(rightSensor)
        centerConc = self.foodConcentration(centerSensor)

        spikes = np.zeros(len(self.sensory_neurons))

        #
        # Chemosensory neurons
        #

        if "ASEL" in self.sensoryMap:
            spikes[self.sensoryMap["ASEL"]] = max(0, leftConc - rightConc)

        if "ASER" in self.sensoryMap:
            spikes[self.sensoryMap["ASER"]] = max(0, rightConc - leftConc)

        if "AWCL" in self.sensoryMap:
            spikes[self.sensoryMap["AWCL"]] = leftConc

        if "AWCR" in self.sensoryMap:
            spikes[self.sensoryMap["AWCR"]] = rightConc

        if "AWAL" in self.sensoryMap:
            spikes[self.sensoryMap["AWAL"]] = centerConc

        if "AWAR" in self.sensoryMap:
            spikes[self.sensoryMap["AWAR"]] = centerConc

        return spikes

    def decodeMotorActivity(self, motorSpikes):
        left = 0.0
        right = 0.0
        if len(self.leftMotors):
            left = np.mean([
                motorSpikes[idx]
                for idx in self.leftMotors
            ])
        if len(self.rightMotors):
            right = np.mean([
                motorSpikes[idx]
                for idx in self.rightMotors
            ])
        return left, right

    def step(self, motorSpikes):
        leftDrive, rightDrive = self.decodeMotorActivity(motorSpikes)
        turnRate = (rightDrive - leftDrive)
        self.heading += (turnRate * 0.4)
        speed = max(0.25, (leftDrive + rightDrive) / 2)
        self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * speed
        self.position = np.clip(self.position, 0, self.world_size)
        concentration = self.foodConcentration(self.position)
        reward = (concentration - self.previousConcentration)
        self.previousConcentration = concentration
        reachedFood = (np.linalg.norm(self.position - self.food_position) < 10)
        if reachedFood:
            reward += 1000
        observation = self.observe()
        return observation, reward, reachedFood