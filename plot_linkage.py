import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import *


################################# LOADING L=100 ##################################
L=100
alphabar = 1/L
eta=1.2

################################## LOADING ########################################
burn_in = 1/10 #We consider that the system reaches stationarity after a fraction burn_in of the time
########## LOADING N=100 #########
N=100
nbpoints=20

omega = np.zeros(nbpoints)
list_means = np.zeros(nbpoints)
list_varmeans = np.zeros(nbpoints)
list_meanvarsX = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX = np.zeros(nbpoints) #variance of the previous item
list_meanvars = np.zeros(nbpoints) #trait variance with linkage
list_varvars = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))

for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  omega[k] = tmp[0]
  list_means[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans[k] = np.var(tmp[3][-postburnin:])
  list_meanvarsX[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX[k] = np.var(tmp[4][-postburnin:])
  list_meanvars[k] = np.mean(tmp[5][-postburnin:])
  list_varvars[k] = np.var(tmp[5][-postburnin:])

gamma = N*alphabar**2/omega**2

########## LOADING N=1000 #########
N2=1000

omega2 = np.zeros(nbpoints)
list_means2 = np.zeros(nbpoints)
list_varmeans2 = np.zeros(nbpoints)
list_meanvarsX2 = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX2 = np.zeros(nbpoints) #variance of the previous item
list_meanvars2 = np.zeros(nbpoints) #trait variance with linkage
list_varvars2 = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L)+"_N"+str(N2)+"/"+str(k)+".npy",allow_pickle=True)
  omega2[k] = tmp[0]
  list_means2[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans2[k] = np.var(tmp[3][-postburnin:])
  list_meanvarsX2[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX2[k] = np.var(tmp[4][-postburnin:])
  list_meanvars2[k] = np.mean(tmp[5][-postburnin:])
  list_varvars2[k] = np.var(tmp[5][-postburnin:])

tmp = None

gamma2 = N2*alphabar**2/omega2**2

################### LOADING L=1000 ################
L3=1000
alphabar3 = 1/L3

omega3 = np.zeros(nbpoints)
list_means3 = np.zeros(nbpoints)
list_varmeans3 = np.zeros(nbpoints)
list_meanvarsX3 = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX3 = np.zeros(nbpoints) #variance of the previous item
list_meanvars3 = np.zeros(nbpoints) #trait variance with linkage
list_varvars3 = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("sims_L"+str(L3)+"_N"+str(N)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L3)+"_N"+str(N)+"/"+str(k)+".npy",allow_pickle=True)
  omega3[k] = tmp[0]
  list_means3[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans3[k] = np.var(tmp[3][-postburnin:])
  list_meanvarsX3[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX3[k] = np.var(tmp[4][-postburnin:])
  list_meanvars3[k] = np.mean(tmp[5][-postburnin:])
  list_varvars3[k] = np.var(tmp[5][-postburnin:])

tmp = None

gamma3 = N*alphabar3**2/omega3**2


################### LOADING L=400, N=500 ################
L4=400
N4=500
alphabar4 = 1/L4

omega4 = np.zeros(nbpoints)
list_means4 = np.zeros(nbpoints)
list_varmeans4 = np.zeros(nbpoints)
list_meanvarsX4 = np.zeros(nbpoints) #L E[alpha**2 X(1-X)]
list_varvarsX4 = np.zeros(nbpoints) #variance of the previous item
list_meanvars4 = np.zeros(nbpoints) #trait variance with linkage
list_varvars4 = np.zeros(nbpoints) #trait variance variance with linkage

tmp = np.load("sims_L"+str(L4)+"_N"+str(N4)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints):
  tmp = np.load("sims_L"+str(L4)+"_N"+str(N4)+"/"+str(k)+".npy",allow_pickle=True)
  omega4[k] = tmp[0]
  list_means4[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans4[k] = np.var(tmp[3][-postburnin:])
  list_meanvarsX4[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX4[k] = np.var(tmp[4][-postburnin:])
  list_meanvars4[k] = np.mean(tmp[5][-postburnin:])
  list_varvars4[k] = np.var(tmp[5][-postburnin:])

tmp = None

gamma4 = N4*alphabar4**2/omega4**2


################### LOADING L=400, N=800 ################
L5=400
N5=800
alphabar5 = 1/L5
nbpoints5 = 10

omega5 = np.zeros(nbpoints5)
list_means5 = np.zeros(nbpoints5)
list_varmeans5 = np.zeros(nbpoints5)
list_meanvarsX5 = np.zeros(nbpoints5) #L E[alpha**2 X(1-X)]
list_varvarsX5 = np.zeros(nbpoints5) #variance of the previous item
list_meanvars5 = np.zeros(nbpoints5) #trait variance with linkage
list_varvars5 = np.zeros(nbpoints5) #trait variance variance with linkage

tmp = np.load("sims_L"+str(L5)+"_N"+str(N5)+"/"+str(0)+".npy",allow_pickle=True)
postburnin = int(len(tmp[3])*(1-burn_in))
for k in range(nbpoints5):
  tmp = np.load("sims_L"+str(L5)+"_N"+str(N5)+"/"+str(k)+".npy",allow_pickle=True)
  omega5[k] = tmp[0]
  list_means5[k] = eta-np.mean(tmp[3][-postburnin:])
  list_varmeans5[k] = np.var(tmp[3][-postburnin:])
  list_meanvarsX5[k] = np.mean(tmp[4][-postburnin:])
  list_varvarsX5[k] = np.var(tmp[4][-postburnin:])
  list_meanvars5[k] = np.mean(tmp[5][-postburnin:])
  list_varvars5[k] = np.var(tmp[5][-postburnin:])

tmp = None

gamma5 = N5*alphabar5**2/omega5**2

##### PLOTTING LINKAGE #####
ax = plt.axes()
ax.plot(gamma,list_meanvars-2*list_meanvarsX,ls="",marker="v",color="blue",label="N="+str(N)+", L="+str(L))
ax.plot(gamma2,list_meanvars2-2*list_meanvarsX2,ls="",marker="^",color="purple",label="N="+str(N2)+", L="+str(L))
ax.plot(gamma3,list_meanvars3-2*list_meanvarsX3,ls="",marker="<",color="darkgreen",label="N="+str(N)+", L="+str(L3))
ax.plot(gamma4,list_meanvars4-2*list_meanvarsX4,ls="",marker=">",color="red",label="N="+str(N4)+", L="+str(L4))
ax.plot(gamma5,list_meanvars5-2*list_meanvarsX5,ls="",marker="o",color="orange",label="N="+str(N5)+", L="+str(L5))
ax.set_xscale("log")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$V_z-\sigma^2$")
ax.legend()
plt.show()
#Plotting Delta/sigma
ax=plt.axes()
ax.loglog(gamma,list_means/np.sqrt(list_meanvars),ls="",marker="v",color="blue",label="N="+str(N)+", L="+str(L))
ax.loglog(gamma2,list_means2/np.sqrt(list_meanvars2),ls="",marker="^",color="purple",label="N="+str(N2)+", L="+str(L))
ax.loglog(gamma3,list_means3/np.sqrt(list_meanvars3),ls="",marker="<",color="darkgreen",label="N="+str(N)+", L="+str(L3))
ax.loglog(gamma4,list_means4/np.sqrt(list_meanvars4),ls="",marker=">",color="red",label="N="+str(N4)+", L="+str(L4))
ax.loglog(gamma5,list_means5/np.sqrt(list_meanvars5),ls="",marker="o",color="orange",label="N="+str(N5)+", L="+str(L5))
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\Delta/\sqrt{V_z}$")
ax.legend()
plt.show()

