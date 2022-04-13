import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/aboulin/')
import matplotlib.pyplot as plt
from COPPY.coppy.rng import evd, monte_carlo
from scipy.stats import norm
from tqdm import tqdm

np.random.seed(42)

lmbd = np.linspace(1,199,199) / 200
n_iter, n_sample, corr = 300, 1024, True
theta, psi1, psi2, Sigma = 2/5, 0.1, 1.0, np.matrix([[1.0,0.8],[0.8,1.0]]) # Sigma = (2*theta)**2 * np.matrix([[0,1.0],[1.0,0]]) for Husler, np.matrix([[1.0,0.8],[0.8,1.0]]) for tEV
asy = [1-psi1, 1-psi2, [psi1, psi2]]
P = np.array([[0.75,0.75*0.75],[0.75*0.75,0.75]])

copula = evd.Asymmetric_logistic(theta = theta, asy = asy,d = 2, Sigma = Sigma, n_sample = n_sample)
copula_miss = evd.Logistic(theta = 1.0, n_sample = n_sample, d = 2)
missing = monte_carlo.Monte_Carlo(n_sample = n_sample, copula = copula,copula_miss = copula_miss, P = P)
scaled = []
for m in tqdm(range(0,n_iter)):
    output = []
    for lmbd_ in lmbd:
        data = copula.sample([norm.ppf])
        miss = missing._gen_missing()
        w = np.array([lmbd_, 1-lmbd_])
        #w = simplex(d = 2)[0]
        wmado_object = monte_carlo.Monte_Carlo(n_sample = n_sample, P = P, w = w, copula = copula)
        value_ = wmado_object._wmado(data, miss, corr = corr)
        output.append(np.sqrt(n_sample)*(value_ - copula.true_wmado(w)))
    scaled.append(output)

scaled = pd.DataFrame(scaled)
print(np.var(scaled))

scaled.to_csv('asy_log_outside_True.csv')
