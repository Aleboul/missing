import numpy as np
import sys
sys.path.insert(0, '/home/aboulin/')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from scipy.stats import norm

from COPPY.coppy.rng import evd

name = "tEV"

lmbd = np.linspace(1,199,199) / 200
n_iter, n_sample, corr = 300, 1024, True
theta, psi1, psi2, Sigma = 0.8, 0.2, 1.0, np.matrix([[1.0,0.8],[0.8,1.0]]) # Sigma = (2*theta)**2 * np.matrix([[0,1.0],[1.0,0]]) for Husler, np.matrix([[1.0,0.8],[0.8,1.0]]) for tEV
P = np.array([[0.75,0.75*0.75],[0.75*0.75,0.75]])
asy = [1-psi1, 1-psi2, [psi1, psi2]]


copula = evd.tEV(theta = theta, psi1 = psi1,d = 2, Sigma = Sigma, n_sample = n_sample)

scaled = pd.read_csv(name+'_True.csv', index_col = 0)

var = []
for i in range(len(lmbd)):
    value_ = [ np.var(scaled.iloc[:,i]),
             copula.var_mado(w = np.array([lmbd[i],1-lmbd[i]]), P = P, p = 0.75 * 0.75, 
             corr = corr), lmbd[i]]
    var.append(value_)

df = pd.DataFrame(var)
df.columns = ["empirical","theoretical", "lambda"]

print(df)

crest = sns.color_palette('crest', as_cmap = True)

fig, ax = plt.subplots()
sns.lineplot(data = df, x = "lambda", y = "theoretical", color = "forestgreen", lw = 1)
sns.scatterplot(data = df, x = "lambda", y = "empirical", color = "forestgreen", s = 10, alpha = 0.5)
corr = False

scaled = pd.read_csv(name+'_False.csv', index_col = 0)

var = []
for i in range(len(lmbd)):
    value_ = [np.var(scaled.iloc[:,i]), 
             copula.var_mado(w = np.array([lmbd[i],1-lmbd[i]]), P = P, p = 0.75 * 0.75, 
             corr = corr), lmbd[i]]
    var.append(value_)

df = pd.DataFrame(var)
df.columns = ["empirical", "theoretical", "lambda"]

print(df)

OrRd = sns.color_palette('OrRd', as_cmap = True)

sns.lineplot(data = df, x = "lambda", y = "theoretical", color = 'darkorange', lw = 1)
sns.scatterplot(data = df, x = "lambda", y = "empirical", color = 'darkorange', s = 10, alpha = 0.5)
ax.set_ylabel(r'$\mathcal{E}^{\mathcal{H}}_n / \mathcal{E}^{\mathcal{H}*}_n$')
ax.set_xlabel(r'$w$')
plt.savefig(name +".pdf")
