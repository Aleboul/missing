import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

d_ = [5,10,15,20,25,30,35,40]
df_1 = pd.read_csv("sup_5_40_256.csv", index_col = 0)
df_1['n'] = 256
df_2 = pd.read_csv("sup_5_40_512.csv", index_col = 0)
df_2['n'] = 512
df_3 = pd.read_csv("sup_5_40_1024.csv", index_col = 0)
df_3['n'] = 1024

df = pd.concat([df_1, df_2, df_3], axis = 0)
#DF = pd.DataFrame()
#
## reshape data
#for d in d_:
#    DF[d] = np.array(df[df['d'] == d]['w_mado'])

#median = DF.median()
fig, ax = plt.subplots()
#ax.plot(d_, median, linewidth = 1, c = 'orange')
#ax.boxplot(np.array(DF), positions = d_, showfliers= False)
ax = sns.boxplot(x = "d", y = "w_mado", data = df, showfliers= False, hue = 'n', palette = 'crest', linewidth = 1)
ax.set_xlabel('d')
ax.set_ylabel(r'$\delta_n^{\mathcal{H}}$')

plt.savefig("boxplot.pdf")