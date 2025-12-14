from SNN import SNN

steps = 1000
i = 5
h = 3
o = 2
pat = [1, 0, 0, 1, 0]
model = SNN(steps, i, h, o, pat)
model.train()