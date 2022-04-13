import numpy as np
import sys
sys.path.insert(0, '/home/aboulin/')
import pandas as pd
from COPPY.coppy.rng import evd, utils

np.random.seed(42)
lmbd = utils.simplex(n = 200, d = 3)

name = "Log_3d"
n_iter, n_sample, corr = 100, 512, True
theta = 0.5
#theta = [0.6,0.5,0.8,0.3]
#asy = [0.4,0.1,0.6,[0.3,0.2], [0.1,0.1], [0.4,0.1], [0.2,0.3,0.2]]
P = np.array([[0.9,0.81,0.81],[0.81,0.9,0.81],[0.81,0.81,0.9]])

copula = evd.Logistic(theta = theta, d = 3, n_sample = n_sample)

scaled = pd.read_csv(name+'_True.csv', index_col = 0)

MISE = []

indices = np.arange(n_iter)
subpart = np.split(indices,10)

varmad = []
for i in range(0,199):
    varmad.append(copula.var_mado(w = np.array(lmbd[i]), P = P, p = 0.9*0.9*0.9, corr = corr))

MISE_VAR = []
for sub in subpart:
    output= []
    for i in range(0,199):
        value = np.power(np.var(scaled.iloc[sub,i]) - varmad[i], 2 )
        print(np.var(scaled.iloc[sub,i]), varmad[i])
        output.append(value)
    MISE_VAR.append(output)

print(np.mean(MISE_VAR))

corr = False

scaled = pd.read_csv(name+'_False.csv', index_col = 0)

varmad = []
for i in range(0,199):
    varmad.append(copula.var_mado(w = np.array(lmbd[i]), P = P, p = 0.9*0.9*0.9, corr = corr))

MISE_VAR = []
for sub in subpart:
    output= []
    for i in range(0,199):
        value = np.power(np.var(scaled.iloc[sub,i]) - varmad[i], 2 )
        output.append(value)
    MISE_VAR.append(output)

print(np.mean(MISE_VAR))