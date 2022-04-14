import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from k_means_constrained import KMeansConstrained

## Colab link :  https://colab.research.google.com/drive/1RKBwvdWpy1auJvKaF5CkRhPlDQM0cxGw#scrollTo=ZMo3qLWTOhiS

"""
    Download Data
"""

data = pd.read_csv('../data/data.csv', index_col=0)
coord = pd.read_csv("../data/coordinates.csv", index_col = 0)

gdf = gpd.read_file("../data/canada.geo.json")

"""
    Print map
"""
#polygon = Polygon([(-82,43), (-82,52), (-51,52), (-51,43)])

fig, ax = plt.subplots()
gdf.plot(linewidth= 0.4, ax =ax, color = "oldlace", edgecolor = "wheat")
ax.scatter(coord['longitude'], coord['latitude'], s = 5, edgecolor = 'darkblue', color = 'wheat')
plt.gca().add_patch(Rectangle((-82,43),31,9,
                    edgecolor='gray',
                    facecolor='none',
                    lw=0.4))
#poly_gdf.boundary.plot(ax = ax, color ='gray')
plt.savefig("map.pdf")

fig, ax = plt.subplots()
gdf.plot(linewidth= 0.4, ax =ax, color = "oldlace", edgecolor = "wheat")
ax.scatter(coord['longitude'], coord['latitude'], s = 5, edgecolor = 'darkblue', color = 'wheat')
ax.set_xlim(-82, -51)
ax.set_ylim(43, 52)
plt.savefig("map_zoom.pdf")

"""
    Selection of data
"""

data = data.drop(index = 29, axis = 0)
data = data.drop(index = 7, axis = 0)
data = data.drop(index = 14, axis = 0)
data = data.drop(index = 6, axis = 0)

sort_data = data.isnull().sum(axis = 1).sort_values()

nb = 91 # 90
size = 7 # 
nb_clst = nb // size

x = np.array(sort_data[0:nb].index)
print(x)

cmap = sns.color_palette("Paired")
X = coord.loc[x]

clf = KMeansConstrained(
     n_clusters= nb_clst,
     size_min=size,
     size_max=size,
     random_state=0
)

clf.fit_predict(X)

"""
    Compute extremal coefficient
"""

def _ecdf(data, miss):
    """Compute uniform ECDF.

        Inputs
        ------
            data (list([float])) : array of observations.

        Output
        ------
            Empirical uniform margin.
    """
    index = np.argsort(data)
    ecdf  = np.zeros(len(index))
    for i in index:
        ecdf[i] = (1.0 / (1.0+np.sum(miss))) * np.sum((data <= data[i]) * miss)
    return ecdf

def _wmado(X, miss, w,corr = {False, True}) :
    """
        This function computes the w-madogram

        Inputs
        ------
        X (array([float]) of n_sample \times d) : a matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram
        
        Outputs
        -------

        w-madogram
    """
    Nnb = X.shape[1]
    Tnb = X.shape[0]
    V = np.zeros([Tnb, Nnb])
    cross = np.ones(Tnb)
    for j in range(0, Nnb):
        
        cross *= miss[:,j]
        X_vec = np.array(X[:,j])
        Femp = _ecdf(X_vec, miss[:,j])
        V[:,j] = np.power(Femp, 1/w[j])
    V *= cross.reshape(Tnb,1)
    if corr == True:
        value_1 = np.amax(V,1)
        value_2 = (1/Nnb) * np.sum(V, 1)
        value_3 = (Nnb - 1)/Nnb * np.sum(V*w,1)
        return (1/np.sum(cross)) * np.sum(value_1 - value_2 - value_3) + ((Nnb-1)/Nnb)*np.sum(w * w/(1+w))
    else :
        value_1 = np.amax(V,1)
        value_2 = (1/Nnb) * np.sum(V, 1)
        mado = (1/(np.sum(cross))) * np.sum(value_1 - value_2)
    return mado

def Pickands(X, miss, w, corr = True):
    Nnb = X.shape[1]
    mado = _wmado(X,miss, w, corr)
    c = (1/Nnb)*np.sum(np.divide(w, 1 + np.array(w)))
    value = (mado + c) / (1-mado-c)
    return value

data_array = data.loc[x,:]
print(data_array)
print(clf.labels_)

output = []
for label in np.unique(clf.labels_):
    group = np.array(np.where(clf.labels_ == label))[0]
    group = np.array(data_array.index[group])
    sub_data = np.transpose(np.array(data_array.loc[group,:]))
    miss = 1 * (sub_data >= 0)
    sum_cross = len(np.where(miss.sum(axis = 1) == size)[0]) # sum(cross), common data
    d = sub_data.shape[1]
    w = np.repeat(1/d, d)
    if sum_cross >= 10.0:
        value = d*Pickands(sub_data, miss, w = w, corr = True)
        output.append([group, value])

output = pd.DataFrame(output)
output.columns = ['cluster', 'extremal_coeff']
print(output)


fig, ax = plt.subplots()
gdf.plot(linewidth= 0.4, ax =ax, color = "oldlace", edgecolor = "wheat")
for i,cluster in enumerate(output['cluster']):
    for index in cluster:
        ax.scatter(X.loc[index][1], X.loc[index][0], edgecolor = cmap[i], s = 5, color = 'wheat')

ax.set_xlim(-82, -51)
ax.set_ylim(43, 52)
plt.savefig('cslt.pdf')

df = []
for i,cluster in enumerate(output['cluster']):
    for index in cluster:
        df.append([index,round(output['extremal_coeff'][i],2), coord.loc[index][0], coord.loc[index][1]])

df = pd.DataFrame(df)
df.columns = ['index', 'ext_coeff', 'latitude', 'longitude']
#df.ext_coeff = (df.ext_coeff / size) * 100

df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.longitude, df.latitude))
df = df.set_crs(4326, allow_override = True)
fig, ax = plt.subplots()
ax.set_xlim(-75, -66)
ax.set_ylim(45, 49)
c = sns.color_palette('OrRd', n_colors = len(df))
gdf.plot(linewidth= 0.8, ax =ax, color = "lightsteelblue", edgecolor = "grey", alpha = 0.5)
df.plot(ax = ax,column = "ext_coeff",cmap = 'OrRd',s = 25, alpha = 0.7, categorical = True,legend = True)
plt.savefig("ext_coeff.pdf")

