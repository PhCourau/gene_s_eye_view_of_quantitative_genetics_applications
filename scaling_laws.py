import numpy as np
import matplotlib.pyplot as plt
from simulate_population import Simulate
#----- Fixed parameters
theta = (.1,.2)
eta = 1/3
T=12

#------ Tested parameters
N=1000
L=100
nbpoints = 10

list_omega = np.logspace(np.log10(N/L)/2+1/10,np.log10(N/L**2)/2-1/10,nbpoints)

list_sigma = np.zeros(nbpoints) #Trait variance
list_delta = np.zeros(nbpoints) #Distance to optimum
list_nu = np.zeros(nbpoints) #Fluctuations of the trait
for (k,omega) in enumerate(list_omega):
	Tomega = T*(omega/list_omega[0]) #adapt duration of the simulation to
					 #omega: no need to go long for strong
					 #selection.
	list_frequencies,list_meantraits,list_vartraits = Simulate(theta,
								omega,
								eta,
								N,
								L,
								Tomega)
	list_sigma[k] = np.mean(list_vartraits[-(T*N)//10:])
	list_delta[k] = np.mean(list_meantraits[-(T*N)//10:])-eta
	list_nu[k] = np.var(list_meantraits[-(T*N)//10:])

plt.loglog(list_omega,list_sigma/list_omega,"o")
plt.show()
plt.loglog(list_omega,np.abs(list_delta)/list_sigma,"o")
plt.show()
plt.loglog(list_omega,list_nu/list_sigma,"o")
plt.show()
