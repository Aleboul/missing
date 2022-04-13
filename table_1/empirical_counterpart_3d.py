import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/aboulin/')
import matplotlib.pyplot as plt
from COPPY.coppy.rng import evd, monte_carlo, utils
from scipy.stats import norm
from tqdm import tqdm

np.random.seed(42)

lmbd = utils.simplex(n = 200, d = 3)
n_iter, n_sample, corr = 300, 512, False

theta = 0.5
theta = [0.6,0.5,0.8,0.3]
asy = [0.4,0.1,0.6,[0.3,0.2], [0.1,0.1], [0.4,0.1], [0.2,0.3,0.2]]
P = np.array([[0.9,0.81,0.81],[0.81,0.9,0.81],[0.81,0.81,0.9]])

copula = evd.Asymmetric_logistic(theta = theta, asy = asy,d = 3, n_sample = n_sample)
copula_miss = evd.Logistic(theta = 1.0, n_sample = n_sample, d = 3)
missing = monte_carlo.Monte_Carlo(n_sample = n_sample, copula = copula,copula_miss = copula_miss, P = P)
scaled = []
for m in tqdm(range(0,n_iter)):
    output = []
    for lmbd_ in lmbd:
        data = copula.sample([norm.ppf])
        miss = missing._gen_missing()
        #w = np.array([lmbd_, 1-lmbd_])
        #w = simplex(d = 2)[0]
        wmado_object = monte_carlo.Monte_Carlo(n_sample = n_sample, P = P, w = lmbd_, copula = copula)
        value_ = wmado_object._wmado(data, miss, corr = corr)
        output.append(np.sqrt(n_sample)*(value_ - copula.true_wmado(lmbd_)))
    scaled.append(output)

scaled = pd.DataFrame(scaled)

scaled.to_csv("Asym_log_False.csv")
