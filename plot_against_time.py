import numpy as np
import matplotlib.pyplot as plt

nb_simus = 19
eta = 1.1
omega,list_traitmean,list_traitvar = [[0]*nb_simus,[0]*nb_simus,[0]*nb_simus]
for k in range(nb_simus):
	omega[k],final_dist,alpha,list_traitmean[k],list_traitvar[k] = np.load(
			"sims/"+str(k)+".npy",allow_pickle=True)

listcolors = plt.get_cmap("viridis")
plot_every = 1
#Plot trait means
for k in range(nb_simus//plot_every):
	plt.plot(list_traitmean[plot_every*k],
		alpha=5/nb_simus,
		color=listcolors(plot_every*k/nb_simus))
plt.show()

#Plot trait variance
for k in range(nb_simus//plot_every):
	plt.plot(list_traitvar[plot_every*k],
			alpha=5/nb_simus,
			color=listcolors(plot_every*k/nb_simus))
plt.show()


