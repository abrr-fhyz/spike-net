import numpy as np
import csv
from collections import OrderedDict
from NeuralNet.Neuron import Neuron
from NeuralNet.Synapse import Synapse

sensoryNeurons = []
interNeurons = []
motorNeurons = []

inputHiddenSynapses = {}
hiddenOutputSynapses = {}
interInputSynapses = {}
interHiddenSynapses = {}
interOutputSynapses = {}
hiddenInputSynapses = {}
outputHiddenSynapses = {}

neuronMap = {}

SENSORY_PREFIXES = ['ADF', 'ADL', 'AQR', 'ASE', 'ASG', 'ASH', 'ASI', 'ASJ', 'ASK', 'AWA', 'AWB', 'AWC', 'BAG', 'CEP', 'FLP', 'IL1', 'IL2', 'OLL', 'OLQ', 'PHA', 'PHB', 'PHC', 'PLM', 'PLN', 'URA', 'URB', 'URX', 'URY', 'AVM', 'ALM', 'ALN', 'PDE']
MOTOR_PREFIXES = ['DA', 'DB', 'DD', 'VD', 'VA', 'VB', 'VC', 'AS']

sensoryCount = 0
interCount = 0
motorCount = 0
exp = 0

def classifyNeuron(name: str):
    for prefix in MOTOR_PREFIXES:
        if name.startswith(prefix):
            if len(name) > len(prefix):
                nextChar = name[len(prefix)]
                if nextChar.isdigit():
                    return 'Motor'
    
    for prefix in SENSORY_PREFIXES:
        if name.startswith(prefix):
            return 'Sensory'
    
    return 'Inter'

def loadConnectome(filepath):
    global sensoryNeurons, interNeurons, motorNeurons, neuronMap, sensoryCount, motorCount, interCount
    
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        
        header = next(reader)
        neuronNames = header[1:]
        
        print(f"Total neurons found: {len(neuronNames)}")
        
        for name in neuronNames:
            neuron = Neuron()
            neuron_type = classifyNeuron(name)
            neuron.setName(name, neuron_type)
            if neuron_type == 'Sensory':
                neuron.identifier(sensoryCount)
                sensoryNeurons.append(neuron)
                sensoryCount += 1
            elif neuron_type == 'Motor':
                neuron.identifier(motorCount)
                motorNeurons.append(neuron)
                motorCount += 1
            else:
                neuron.identifier(interCount)
                interNeurons.append(neuron)
                interCount += 1
            neuronMap[name] = neuron
        
        print(f"Sensory neurons: {len(sensoryNeurons)}")
        print(f"Interneurons: {len(interNeurons)}")
        print(f"Motor neurons: {len(motorNeurons)}")
        
        synapses = []
        synapse_count = 0
        
        for row in reader:
            pre_neuron_name = row[0]
            pre_neuron = neuronMap[pre_neuron_name]
            
            for idx, weight_str in enumerate(row[1:]):
                weight = float(weight_str)
                
                if weight > 0:
                    post_neuron_name = neuronNames[idx]
                    post_neuron = neuronMap[post_neuron_name]
                    
                    synapse = Synapse(weight, pre_neuron, post_neuron)
                    synapses.append(synapse)
                    synapse_count += 1
        
        print(f"Total synapses created: {synapse_count}")
        
        return synapses

def classifySynapses(synapses: list[Synapse]):
    global inputHiddenSynapses, interInputSynapses, interHiddenSynapses, hiddenOutputSynapses, interOutputSynapses, hiddenInputSynapses, outputHiddenSynapses, exp
    for synapse in synapses:
        x = synapse.preSpike
        y = synapse.postSpike
        if x.neuronType == 'Sensory' and y.neuronType == 'Inter':
            inputHiddenSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Sensory' and y.neuronType == 'Sensory':
            interInputSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Inter' and y.neuronType == 'Motor':
            hiddenOutputSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Inter' and y.neuronType == 'Inter':
            interHiddenSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Motor' and y.neuronType == 'Motor':
            interOutputSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Inter' and y.neuronType == 'Sensory':
            hiddenInputSynapses[(x.idx, y.idx)] = synapse
        elif x.neuronType == 'Motor' and y.neuronType == 'Inter':
            outputHiddenSynapses[(x.idx, y.idx)] = synapse
        else:
            exp += 1
            #print(f"Exception found -> {x.neuronType} to {y.neuronType}")

def buildNervousSystem():
    synapses = loadConnectome('Data/data.csv')
    print("========== 302 NEURONS INITIALISED =========")
    return synapses, sensoryNeurons, interNeurons, motorNeurons

def buildSynapticStructure(synapses):
    classifySynapses(synapses)
    print("========== 5149 SYNAPSES INITIALISED =========")
    return inputHiddenSynapses, hiddenOutputSynapses, interInputSynapses, interHiddenSynapses, interOutputSynapses, outputHiddenSynapses, hiddenInputSynapses
