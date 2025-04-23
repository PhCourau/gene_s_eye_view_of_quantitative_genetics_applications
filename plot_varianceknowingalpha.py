# The goal is here to use the empirical distribution of the alphabar instead of the predicted theoretical one.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import *

theta2N= (.1,.2)
eta=1.2
L=100
N=1000
alphabar = 1/L
#Exponentially distributed genetic effects
list_alpha = np.linspace(0,6/L,100) #numerical instabilities if we allow alpha too large
list_proba_alpha = L*np.exp(-L*list_alpha)
list_proba_alpha /= np.sum(list_proba_alpha)
klim = 81 #Degree of precision for the theoretical computations under strong selection
#If klim is too low then the genetic variance sigma2_th_ss will have a bump for strong selection
listcolors=plt.get_cmap("viridis")
transparency=1

################################## LOADING ########################################
burn_in = 1/10 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=100 #########
nbpoints=10
real_alpha = np.zeros((nbpoints,L)) #numerical instabilities if we allow alpha too large

omega = np.zeros(nbpoints)
list_means = np.zeros(nbpoints)
list_varmeans = np.zeros(nbpoints)
list_meanvarsX = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX = np.zeros(nbpoints) #variance of the previous item
list_meanvars = np.zeros(nbpoints) #trait variance with linkage
list_varvars = np.zeros(nbpoints) #trait variance variance with linkage
list_rho = np.zeros(nbpoints) #Autocorrelation parameters

tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  omega[k] = tmp[0]
  real_alpha[k] = tmp[2]
  list_means[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans[k] = np.var(tmp[3][-postburnin:])
  list_rho[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/list_varmeans[k])
  list_meanvarsX[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX[k] = np.var(tmp[4][-postburnin:])
  list_meanvars[k] = np.mean(tmp[5][-postburnin:])
  list_varvars[k] = np.var(tmp[5][-postburnin:])

gamma = N*alphabar**2/omega**2

###Selection coefficients
#Weak selection
s2N_weak = np.zeros(nbpoints)
for k in range(nbpoints):
  s2N_weak[k] = find_s2N_weak(gamma[k],theta2N,eta,L,alphabar,list_alpha,list_proba_alpha)

list_delta_weak = -omega**2 * s2N_weak/(2*N) #Recall s = s2N/(2*N)

#Moderate selection
s2N_moderate = find_s2N_moderate(theta2N,eta,L,list_alpha,list_proba_alpha)

list_delta_moderate = -omega**2 * s2N_moderate/(2*N)

#Strong selection
s2N_strong = np.array([find_s2N_strong(gamma[k],
                                     theta2N,
                                     eta,
                                     L,
                                     alphabar,
                                     list_alpha,
                                     list_proba_alpha,
                                     klim=klim)
             for om in omega])
list_delta_strong = -omega**2 * s2N_strong/(2*N)

#### Plotting Sigma
sigma2_th = np.array([
                      genetic_variance(sstar,theta2N,L,list_alpha,list_proba_alpha)
                      for (k,sstar) in enumerate(s2N_weak)])

sigma2_th_ss = np.array([
                      genetic_variance_strong(s2N_strong[k],
                                              gamma[k],
                                              theta2N,
                                              L,
                                              alphabar,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=klim)
                      for k in range(nbpoints)])
realsigma2_th = np.array([
                      genetic_variance(sstar,theta2N,L,real_alpha[k],np.ones(L)/L)
                      for (k,sstar) in enumerate(s2N_weak)])

realsigma2_th_ss = np.array([
                      genetic_variance_strong(s2N_strong[k],
                                              gamma[k],
                                              theta2N,
                                              L,
                                              alphabar,
                                              real_alpha[k],
                                              np.ones(L)/L,
                                              klim=klim)
                      for k in range(nbpoints)])

ax=plt.axes()
ax.plot(gamma,np.sqrt(list_meanvars),marker="v",label="N="+str(N),ls="",color="blue",alpha=transparency)
#If we neglect linkage, then we expect Var[z] = 2 L E[alpha**2 X(1-X)] which is why we plot
ax.plot(gamma,np.sqrt(2*list_meanvarsX),marker="1",label="N="+str(N)+" (no linkage)",ls="",color="blue",alpha=transparency)

ax.loglog(gamma,np.sqrt(sigma2_th),label="Weak/moderate selection",color=listcolors(.9),alpha=transparency,linestyle="--")
ax.loglog(gamma,np.sqrt(sigma2_th_ss), label="Strong selection",color=listcolors(.6),alpha=transparency,linestyle="--")
ax.loglog(gamma,np.sqrt(realsigma2_th),label="Weak/moderate selection (known alpha)",color=listcolors(.9),alpha=transparency)
ax.loglog(gamma,np.sqrt(realsigma2_th_ss), label="Strong selection (known alpha)",color=listcolors(.6),alpha=transparency)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\sigma$")
ax.legend()
plt.show()

