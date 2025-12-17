from NeuralNet.SNN import SNN

steps = 1000
i = 6
h = 7
o = 4
pat = [0, 1, 0, 1]
model = SNN(steps, i, h, o, pat)
model.train()