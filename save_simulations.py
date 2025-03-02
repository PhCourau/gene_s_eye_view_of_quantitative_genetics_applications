import numpy as np
from simulate_population import generate_pop, Simulate
#----- Fixed parameters
eta = 1.2
T= 12
N=500
L=100
thetaN = (.1,.2) # The rate of mutation from 0 to +1 is thetaN[0]/N per organism
                 # per generation per locus
nbpoints = 20

#list_omega = np.logspace(np.log10(N/L)/2+1/10,np.log10(N/L**2)/2-1/10,nbpoints)

#It appears omega should be greater than 1e-3 ?
list_omega = np.logspace(np.log10(N/L)/2+1/2,np.log10(N/L**2)/2-1/5,nbpoints)


#The starting population must be close to the optimum otherwise the fitness gets
#degenerate
pop0 = generate_pop((eta,2-eta),N,L)

for (k,omega) in enumerate(list_omega):
	print("Simulation "+str(k)+" of "+str(nbpoints))
	sim = [omega,0,0,0,0]
	sim[1:5] = Simulate(thetaN,omega,eta,N,L,T,initial_pop=pop0)
	sim = np.array(sim,dtype="object")

	np.save("sims/"+str(k)+".npy",sim)
