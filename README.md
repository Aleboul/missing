# Supplementary material

This github repository contains supplementary materials for the simulation study of the paper :

- [Non-parametric estimator of a multivariate madogram for missing-data and extreme value framework](https://arxiv.org/abs/2112.13575)

All codes are designed using Python programming language. Below, one may find a short description for each folder.

## coppy

This folder packs up all the necessary object to sample from copulae. More details can be found in the reference below :

- [COPPY module](https://arxiv.org/abs/2203.17177)

## fig_1

This folder is associated to experiment **E1**.
The script empirical_counterpart.py generates 199 samples from the specified model and estimate the **w**-madogram for 199 values of **w** in ]0,1[. This step are reproduced 300 times and store afterwards results in a csv file.
This csv file contains 300 estimator of the hybrid and corrected estimator of the **w**-madogram for 199 values in [0,1] (a 300 x 199 table).
One can plot the resulting figures in Fig. 1 of the paper by using the plot.py script.

## fig_2_3

Contains all scripts to perform simulation of experiment **E2** for a peculiar grid of the simplex in dimension 3.
The file empirical_counterpart.py is used to compute the empirical counterpart and the theoretical value of the asymptotic variance of **w**-madogram. Results are store in a csv file which is call by plot.py file to produce panel of figures 2 and 3.

## fig_4

inc_dim.py sample 300 points of the simplex. For each drawn point, an empirical counterpart of the asymptotic variance is estimated using 300 samples from a logistic model. We thus compute its relative error. This step is reproduced for several value of d. 
One can find all csv files for different value of n = 256, 512, 1024. Figure 4 is obtained by appealing fig.py.

## fig_5

Contains data (data.csv, canada.geo.json, coordinates.csv) and the script (hybrid.py) to perform application on the paper.

## table_1

empirical_counterpart.py and empirical_counterpart_3d.py estimate **w**-madogram for respectively d=2 and d=3. For d=2, an ordered grid is used to estimate the integral. For d=3, 199 points of the simplex are drawn. All the results are stored in multiple csv files, those are called by compute_mise (for d = 2) and compute_mise_3d (for d = 3) to obtain results of Table 1 of the paper.