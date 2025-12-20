from NeuralNet.SNN import SNN

steps = 100
i = 6
h = 7
o = 4
model = SNN(steps, i, h, o)
model.train()