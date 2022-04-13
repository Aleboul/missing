import numpy as np
import time
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/aboulin/')

from CoPY.src.rng.utils import simplex
from CoPY.src.rng.evd import Logistic
from CoPY.src.rng.monte_carlo import Monte_Carlo

def max(a,b):
    if a >= b:
        return a
    else:
        return b

d_ = [5,10,20,25,30,35,40]
theta = 0.5
n_sample = [256]
n_iter = 300
sup_ = []
t_ = []
w_ = []
array_ = []
for d in d_:
    print(d)
    P = np.ones([d, d])
    p = 1.0
    copula = Logistic(theta= theta, d = d, n_sample = np.max(n_sample))
    t = time.process_time()
    for k in tqdm(range(0,300)):
        w = simplex(d = d)[0]
        Monte = Monte_Carlo(n_iter = n_iter, n_sample = n_sample, w = w, copula = copula, P = P)
        df_wmado = Monte.finite_sample([norm.ppf], corr = True)
        sigma_empi_ = df_wmado['scaled'].var()
        sigma_theo_ = max(0,copula.var_mado(w,P, p, corr = True))
        value_ = np.abs(sigma_empi_ - sigma_theo_) / sigma_theo_
        array_.append(value_)
        w_.append(w)
    t = time.process_time() - t
    t_.append(t)

df = pd.DataFrame()
df['w'] = list(w_)
df['w_mado'] = array_
df.columns = ['w', 'w_mado']
print(df)
df.to_csv("sup_5_40_256.csv")
