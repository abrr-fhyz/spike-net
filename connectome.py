import numpy as np
from NeuralNet.Neuron import Neuron
from NeuralNet.Synapse import Synapse
import csv
from collections import OrderedDict

sensoryNeurons = []
interNeurons = []
motorNeurons = []

def process(line):
    idx = 0
    for key, val in line:
        n = Neuron()
        idx += 1


with open('data.csv', mode = 'r') as file:
    connectome = csv.Reader(file)
    first = True
    for line in connectome:
        if first:
            process(line)
            first = False
        else:
            synaptic(line)