#This is just to test the convergence of E[X(1-X)] to the stationary prediction in
#the neutral case

import numpy as np
import matplotlib.pyplot as plt
from simulate_population import Simulate
from mean_weak_moderate_strong import *

L=100
nbpoints = 2
list_vars = np.zeros(nbpoints)
list_N = np.linspace(1000,5000,nbpoints,dtype="int")
thetaN = (.1,.2)

for (k,N) in enumerate(list_N):
  tmp = Simulate(thetaN,np.inf,L,N,L,10,alphamethod = lambda L:np.ones(L))
  list_vars[k] = np.mean(tmp[-1][-N:]/L)

plt.plot(list_N,list_vars)
plt.plot(list_N,[var_WF_FD(0,0,(.1,.2))]*len(list_N))
plt.show()
