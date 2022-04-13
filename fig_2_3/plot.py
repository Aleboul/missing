crest = sns.color_palette('crest', as_cmap = True)

df_wmado = pd.read_csv("/home/aboulin/Documents/stage/papier/code/multivariate_ed/output/missing/var_mado_sym_FALSE.csv")
print(df_wmado)

W2, W3 = np.meshgrid(W2, W3)
empirical_sigma = np.array(df_wmado['empirical_sigma'])
theoretical_sigma = np.array(df_wmado['theoretical_sigma'])
gap_sigma = np.abs(empirical_sigma - theoretical_sigma)
EMPIRICAL = empirical_sigma.reshape(W2.shape)
THEORETICAL = theoretical_sigma.reshape(W2.shape)
GAP = gap_sigma.reshape(W2.shape)
print(EMPIRICAL.shape)
print(THEORETICAL.shape)
print(W2.shape)
print(W3.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W2, W3, THEORETICAL, alpha = 0.5, color = crest(1.0), rstride=1, cstride=1)
ax.scatter(W2, W3, EMPIRICAL, alpha = 0.5, color = crest(1.0), s = 1)
ax.set_xlabel(r'$w_2$')
ax.set_ylabel(r'$w_3$')
ax.set_zlabel(r"$\mathcal{S}^{\mathcal{H}}_n$")

ax.azim = -110
ax.dist = 8
ax.elev = 15

plt.show()

plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")

#levels = [0.00,0.01,0.02, 0.03, 0.04,0.05]
#fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
#contourf_ = ax[0].contourf(W2, W3, THEORETICAL, 4, cmap = crest, linestyles = 'dashed', levels = levels)
#contourf_ = ax[1].contourf(W2, W3, EMPIRICAL, 4, cmap = crest, linestyles = 'dashed', levels = levels)
#cbar = fig.colorbar(contourf_, ax=ax.ravel().tolist(), shrink=0.95)
#ax[0].set_xlim(0,1)
#ax[0].set_ylim(0,1)
#ax[0].set_xlabel(r'$w_2$')
#ax[0].set_ylabel(r'$w_3$')
#ax[1].set_xlim(0,1)
#ax[1].set_ylim(0,1)
#ax[1].set_xlabel(r'$w_2$')
#ax[1].set_ylabel(r'$w_3$')
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/var_mado.pdf")
