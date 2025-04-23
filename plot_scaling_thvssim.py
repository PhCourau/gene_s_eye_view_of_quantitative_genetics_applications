import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import *

theta2N= (.1,.2)
eta=1.2
L=400
alphabar = 1/L
#Exponentially distributed genetic effects
list_alpha = np.linspace(0,6/L,100) #numerical instabilities if we allow alpha too large
list_proba_alpha = L*np.exp(-L*list_alpha)
list_proba_alpha /= np.sum(list_proba_alpha)
klim = 81 #Degree of precision for the theoretical computations under strong selection (numerical problems if it becomes too large)
#If klim is too low then the genetic variance sigma2_th_ss will have a bump for strong selection

limit_autocor = 2000 #We will compute the correlation between Delta_t and Delta_{t+s} for s smaller
                     #than limit_autocor

# What shall we plot ?
plotdelta = True
plotnu = True
plotsigma = True
plotrho = True
plotautocor = False
listcolors=plt.get_cmap("viridis")


################################## LOADING ########################################
burn_in = 1/10 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=500 #########
N=500
nbpoints=20

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
  list_means[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans[k] = np.var(tmp[3][-postburnin:])
  list_rho[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/list_varmeans[k])
  list_meanvarsX[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX[k] = np.var(tmp[4][-postburnin:])
  list_meanvars[k] = np.mean(tmp[5][-postburnin:])
  list_varvars[k] = np.var(tmp[5][-postburnin:])

gamma = N*alphabar**2/omega**2

########## LOADING N=800 #########
N2=800
nbpoints2=10

omega2 = np.zeros(nbpoints2)
list_means2 = np.zeros(nbpoints2)
list_varmeans2 = np.zeros(nbpoints2)
list_meanvarsX2 = np.zeros(nbpoints2) #L E[alpha**2 X(1-X)]
list_varvarsX2 = np.zeros(nbpoints2) #variance of the previous item
list_meanvars2 = np.zeros(nbpoints2) #trait variance with linkage (if linkage=0, should be 2*list_meanvarsX)
list_varvars2 = np.zeros(nbpoints2) #trait variance variance with linkage
list_rho2 = np.zeros(nbpoints2) #Autocorrelation parameters
list_autocor = np.zeros((nbpoints2,limit_autocor)) #Autocorrelations

tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints2):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(k)+".npy",allow_pickle=True)
  omega2[k] = tmp[0]
  list_means2[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans2[k] = np.var(tmp[3][-postburnin:])
  list_rho2[k] = -np.log(np.cov(tmp[3][-postburnin-1:-1],tmp[3][-postburnin:])[0,1]/list_varmeans2[k])
  list_meanvarsX2[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX2[k] = np.var(tmp[4][-postburnin:])
  list_meanvars2[k] = np.mean(tmp[5][-postburnin:])
  list_varvars2[k] = np.var(tmp[5][-postburnin:])
  if plotautocor:
    list_autocor[k,0] = 1
    for j in range(1,limit_autocor):
      list_autocor[k,j] =np.cov(tmp[3][-postburnin:-j],tmp[3][-postburnin+j:])[0,1]/list_varmeans2[k]

tmp = None

gamma2 = N2*alphabar**2/omega2**2

##### PLOTTING DELTA #####
if plotdelta:
  ax = plt.axes()
  ax.set_xscale("log")
  ax.set_yscale("log")
  line1 = ax.plot(gamma,list_means,"v",label="N="+str(N),color="blue")
  line2 = ax.plot(gamma2,list_means2,"^",label="N="+str(N2),color="purple")

###Selection coefficients
#Weak selection
s2N_weak = np.zeros(nbpoints)
for k in range(nbpoints):
  s2N_weak[k] = find_s2N_weak(gamma[k],theta2N,eta,L,alphabar,list_alpha,list_proba_alpha)

# More precise:
# Based on the fixed point equation:
if plotdelta:
  list_delta_weak = -omega**2 * s2N_weak/(2*N) #Recall s = s2N/(2*N)
  line3 = ax.loglog(gamma,-list_delta_weak,label="Weak selection",color=listcolors(.9))

#Moderate selection
s2N_moderate = find_s2N_moderate(theta2N,eta,L,list_alpha,list_proba_alpha)

if plotdelta:
  list_delta_moderate = -omega**2 * s2N_moderate/(2*N)
  line4 = ax.plot(gamma,-list_delta_moderate,label="Moderate selection",color=listcolors(.75))

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
if plotdelta:
  list_delta_strong = -omega**2 * s2N_strong/(2*N)
  line5 = ax.plot(gamma,-list_delta_moderate,label="Strong selection",color=listcolors(0.6))
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel(r"$\Delta$")
  ax.legend()
  plt.show()

#####   PLOTTING NU #####
if plotnu:
  ax = plt.axes()
  ax.loglog(gamma,np.sqrt(list_varmeans),"v",label="N="+str(N),color="blue")
  ax.loglog(gamma2,np.sqrt(list_varmeans2),"^",label="N="+str(N2),color="purple")
  nu = alphabar/np.sqrt(2*gamma)
  ax.loglog(gamma,nu,label="Prediction",color=listcolors(.6))
  neutral_prediction = (L/N*genetic_variance(0,theta2N,L,list_alpha,list_proba_alpha)
                                        /(theta2N[0]+theta2N[1]))
  ax.loglog(gamma,np.sqrt(1/(1/neutral_prediction+1/nu**2)),label="Prediction (corrected)",color="red")

  ax.legend()
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel(r"$\nu$")
  plt.show()

##### PLOTTING SIGMA #####
if plotsigma:
  ax = plt.axes()
  ax.set_xscale("log")
  ax.set_yscale("log")
  transparency=1
  ax.plot(gamma,np.sqrt(list_meanvars),marker="v",label="N="+str(N),ls="",color="blue",alpha=transparency)
  #If we neglect linkage, then we expect Var[z] = 2 L E[alpha**2 X(1-X)] which is why we plot
  ax.plot(gamma,np.sqrt(2*list_meanvarsX),marker="1",label="N="+str(N)+" (no linkage)",ls="",color="blue",alpha=transparency)
  ax.plot(gamma2,np.sqrt(list_meanvars2),marker="^",label="N="+str(N2),ls="",color="purple",alpha=transparency)
  ax.plot(gamma2,np.sqrt(2*list_meanvarsX2),marker="2",label="N="+str(N2)+" (no linkage)",ls="",color="purple",alpha=transparency)

sigma2_th = np.array([
                      genetic_variance(sstar,theta2N,L,list_alpha,list_proba_alpha)
                      for sstar in s2N_weak])

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

if plotsigma:
  ax.loglog(gamma,np.sqrt(sigma2_th),label="Weak/moderate selection",color=listcolors(.9),alpha=transparency)
  ax.loglog(gamma,np.sqrt(sigma2_th_ss), label="Strong selection",color=listcolors(.6),alpha=transparency)
  ax.loglog(gamma,np.sqrt(sigma2_th_ss*(1-2*sigma2_th_ss*gamma/(alphabar**2 * N))),
            label="Strong selection (corrected for N="+str(N)+")",color="pink",alpha=transparency)
  ax.loglog(gamma,np.sqrt(sigma2_th_ss*(1-2*sigma2_th_ss*gamma/(alphabar**2*N2))),
            label="Strong selection (corrected for N="+str(N2)+")",color="red",alpha=transparency)
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel(r"$\sigma$")
  ax.legend()
  plt.show()

##### PLOTTING LINKAGE #####
ax = plt.axes()
ax.plot(gamma,list_meanvars-2*list_meanvarsX,"o",color="blue")
ax.plot(gamma2,list_meanvars2-2*list_meanvarsX2,"o",color="purple")
ax.set_xscale("log")
plt.show()

##### PLOTTING N RHO/sigma2 #####
if plotrho:
  ax = plt.axes()
  ax.loglog(gamma,N*list_rho/list_meanvars,"v",label="N="+str(N),color="blue")
  ax.loglog(gamma2,N2*list_rho2/list_meanvars2,"^",label="N="+str(N2),color="purple")

  rhosig2 = gamma/(alphabar**2)
  correctedrhosig2 = rhosig2 + (theta2N[0]+theta2N[1])/(2*sigma2_th)
                                #recall |theta| = (theta2N[0]+theta2N[1])/(2*N)


  ax.loglog(gamma,rhosig2,label="Prediction",color=listcolors(0.6))
  ax.loglog(gamma,correctedrhosig2,label="Prediction (corrected)",color="red")
  ax.set_xlabel(r"$\gamma$")
  ax.set_ylabel(r"$\frac{\rho N_e}{\sigma^2}$")
  ax.legend()
  plt.show()

##### PLOTTING AUTOCORRELATION #####
if plotautocor:
  ax=plt.axes()

  for k in np.arange(0,nbpoints2):
    ax.plot(-np.log(list_autocor[k]),"o",color=listcolors(k/nbpoints2))
    ax.plot(list_rho[k]*np.arange(limit_autocor),color=listcolors(k/nbpoints2))

  #Color scale
  gammamin,gammamax=(np.min(np.append(gamma,gamma2)),np.max(np.append(gamma,gamma2)))
  # Normalizer
  norm = mpl.colors.Normalize(vmin=gammamin, vmax=gammamax)
  # creating ScalarMappable
  sm = plt.cm.ScalarMappable(cmap=listcolors, norm=norm)
  sm.set_array([])
  plt.colorbar(sm,ticks=(gammamin,gammamax),format=("%.1e"))

  ax.set_xlabel(r"$n$")
  ax.set_ylabel(r"$\exp(-\rho_n)$")
  plt.show()
