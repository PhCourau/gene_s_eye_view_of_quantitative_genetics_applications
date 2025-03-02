import numpy as np
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import mean_sigma2

thetaN = (.1,.2)
N=1000
T= 10000*N

listX = np.zeros(T)
listX[0] = N//2
for t in range(T-1):
  listX[t+1] = np.random.binomial(N,listX[t]/N)  #Selection
  listX[t+1] += (np.random.binomial(N-listX[t+1],thetaN[0]/N)
                 - np.random.binomial(listX[t+1],thetaN[1]/N)) #Mutation
  if (t*100)%T == 0:
    print(str((t*100)//T)+" per cent completed")

listX /= N
print(np.mean(listX)) #Should be close to .33
print(np.mean(listX * (1-listX))) #Should be close to .083
#Conclusion: both are verified ! The stationary distribution of WF is confirmed.
# The mismatch between trait variance and predicted variance must stem from
# a problem in the simulations


