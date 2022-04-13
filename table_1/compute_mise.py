import numpy as np
import sys
sys.path.insert(0, '/home/aboulin/')
import pandas as pd
from COPPY.coppy.rng import evd

name = "Asy_log"
lmbd = np.linspace(1,199,199) / 200
lmbd = [0.5]
n_iter, n_sample, corr = 300, 1024, True
theta, psi1, psi2, Sigma = 2/5, 0.1, 1.0, (2*1.0)**2 * np.matrix([[0,1.0],[1.0,0]]) # Sigma = (2*theta)**2 * np.matrix([[0,1.0],[1.0,0]]) for Husler, np.matrix([[1.0,0.8],[0.8,1.0]]) for tEV, asy = [1-psi1, 1-psi2, [psi1, psi2]] for asy_log
asy = [1-psi1, 1-psi2, [psi1, psi2]]
P = np.array([[0.75,0.75*0.75],[0.75*0.75,0.75]])

copula = evd.Asymmetric_logistic(theta = theta, asy = asy,Sigma = Sigma, d = 2, n_sample = n_sample)

scaled = pd.read_csv(name+'_True.csv', index_col = 0)

MISE = []

indices = np.arange(n_iter)
subpart = np.split(indices,10)

varmad = []
for i in range(0,199):
    varmad.append(copula.var_mado(w = np.array([lmbd[i],1-lmbd[i]]), P = P, p = 0.75 * 0.75, corr = corr))

MISE_VAR = []
for sub in subpart:
    output= []
    for i in range(0,199):
        value = np.power(np.var(scaled.iloc[sub,i]) - varmad[i], 2 )
        output.append(value)
    MISE_VAR.append(output)

print(np.mean(MISE_VAR))

corr = False

scaled = pd.read_csv(name+'_False.csv', index_col = 0)

varmad = []
for i in range(0,199):
    varmad.append(copula.var_mado(w = np.array([lmbd[i],1-lmbd[i]]), P = P, p = 0.75 * 0.75, corr = corr))

MISE_VAR = []
for sub in subpart:
    output= []
    for i in range(0,199):
        value = np.power(np.var(scaled.iloc[sub,i]) - varmad[i], 2 )
        output.append(value)
    MISE_VAR.append(output)

print(np.mean(MISE_VAR))