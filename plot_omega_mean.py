import numpy as np
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import *

thetaN=(.1,.2)
eta=1.2
N=1000
L=100
nbpoints=100

# What shall we plot ?
plotdelta = True
plotnu = True
plotsigma = True
plotrho = True

omega,list_traitmean,list_traitvar = [np.zeros(nbpoints),[0]*nbpoints,[0]*nbpoints]
for k in range(nbpoints):
  omega[k],final_dist,alphas,list_traitmean[k],list_traitvar[k] = np.load("sims_L100_N1000/"+str(k)+".npy",allow_pickle=True)

burn_in = 3/4 #We consider that the system reaches stationarity after a fraction burn_in of the time-series

postburnin = int(len(list_traitmean[0])*(1-burn_in))
list_means = np.zeros(nbpoints)
list_varmeans = np.zeros(nbpoints)
list_meanvars = np.zeros(nbpoints)
list_rho = np.zeros(nbpoints) #Autocorrelations
for k in range(nbpoints):
  list_means[k] = eta-np.mean(list_traitmean[k][-postburnin:])
  list_varmeans[k] = np.var(list_traitmean[k][-postburnin:])
  list_rho[k] = -np.log(np.cov(list_traitmean[k][-postburnin-1:-1],
                       list_traitmean[k][-postburnin:])[0,1]/list_varmeans[k])
  list_meanvars[k] = np.mean(list_traitvar[k][-postburnin:])


alphabar = 1/L
gamma = N*alphabar**2/omega**2

##### PLOTTING DELTA #####
if plotdelta:
  ax = plt.axes()
  line1 = ax.loglog(omega,list_means,"o",label="Simulations")

###Selection coefficients
#Exponentially distributed genetic effects
list_alpha = np.linspace(0,5/L,100)
list_proba_alpha = L*np.exp(-L*list_alpha)
list_proba_alpha /= np.sum(list_proba_alpha)

#Weak selection
sN_weak = np.zeros(nbpoints)
for k in range(nbpoints):
  sN_weak[k] = find_sN_weak(omega[k]**2,thetaN,eta,N,L,list_alpha,list_proba_alpha)

# More precise:
#list_delta_weak = np.array([
#           mean_phenotype(sstar,thetaN,L,list_alpha,list_proba_alpha)-eta
#           for sstar in sN_weak])
# Based on the fixed point equation:
if plotdelta:
  list_delta_weak = -omega**2 * sN_weak/N
  line2 = ax.loglog(omega,-list_delta_weak,label="Weak selection")

#Moderate selection
sN_moderate = find_sN_moderate(thetaN,eta,L,list_alpha,list_proba_alpha)

if plotdelta:
  list_delta_moderate = -omega**2 * sN_moderate/N
  line3 = ax.loglog(omega,-list_delta_moderate,label="Moderate selection")

#Strong selection
sN_strong = np.array([find_sN_strong(om**2,
                                     thetaN,
                                     eta,
                                     N,
                                     L,
                                     list_alpha,
                                     list_proba_alpha,
                                     klim=40)
             for om in omega])
if plotdelta:
  list_delta_strong = -omega**2 * sN_strong/N
  line4 = ax.loglog(omega,-list_delta_moderate,label="Strong selection")
  plt.title("Delta")
  ax.legend()
  plt.show()

#####   PLOTTING NU #####
if plotnu:
  ax = plt.axes()
  ax.loglog(omega,np.sqrt(list_varmeans),"o")
  nu = alphabar/np.sqrt(2*gamma)
  ax.loglog(omega,nu)
  plt.title("Nu")
  plt.show()

##### PLOTTING SIGMA #####
if plotsigma:
  ax = plt.axes()
  ax.loglog(omega,np.sqrt(list_meanvars),"o")

sigma2_th = np.array([
                      genetic_variance(sstar,thetaN,L,list_alpha,list_proba_alpha)
                      for sstar in sN_weak])

sigma2_th_ss = np.array([
                      genetic_variance_strong(sN_strong[k],
                                              omega[k]**2,
                                              thetaN,
                                              N,
                                              L,
                                              list_alpha,
                                              list_proba_alpha)
                      for k in range(nbpoints)])

if plotsigma:
  ax.loglog(omega,np.sqrt(sigma2_th))
  ax.loglog(omega,np.sqrt(sigma2_th_ss))
  plt.title("Sigma")
  plt.show()


##### PLOTTING RHO #####
if plotrho:
  ax = plt.axes()
  ax.loglog(omega,list_rho,"o",label="Simulations")
  rho_th = gamma/(N*alphabar**2)*sigma2_th
  ax.loglog(omega,rho_th,label="Weak selection")
  rho_th_ss = gamma/(N*alphabar**2)*sigma2_th_ss
  ax.loglog(omega,rho_th_ss,label="Strong selection")
  plt.title("Rho")
  ax.legend()
  plt.show()

##### PLOTTING AUTOCORRELATION #####
listcolors=plt.get_cmap("viridis")

limit_autocor = 1000
for k in np.arange(0,nbpoints,nbpoints//10):
  list_autocor = np.zeros(limit_autocor)
  for j in range(1,limit_autocor+1):
    list_autocor[j-1] = np.cov(list_traitmean[k][-postburnin:-j],
                             list_traitmean[k][-postburnin+j:]
                             )[0,1]/list_varmeans[k]
  plt.plot(list_autocor,color=listcolors(k/nbpoints))
plt.show()
