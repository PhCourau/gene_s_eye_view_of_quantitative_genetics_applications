import numpy as np
import matplotlib.pyplot as plt
from mean_weak_moderate_strong import *

###########First case (theta2N=(.1,.1))#######
#Phase transitions
theta2N= (.1,.1)
eta=1.5
L=100 #The value of L does not matter for this plot
list_alpha = np.linspace(0,5/L,100) #numerical instabilities if we allow alpha too large
list_proba_alpha = L*np.exp(-L*list_alpha)
list_proba_alpha /= np.sum(list_proba_alpha)
alphabar = np.sum(list_alpha*list_proba_alpha)

save = True #Do we want to save the plots ?

sigma2_neutral = genetic_variance(0,theta2N,L,list_alpha,list_proba_alpha)

s2N_moderate = find_s2N_moderate(theta2N,
                                   eta,
                                   L,
                                   list_alpha,
                                   list_proba_alpha,
                                   a0=-10*L,
                                   b0=10*L)

sigma2_moderate = genetic_variance(s2N_moderate,
                                     theta2N,
                                     L,
                                     list_alpha,
                                     list_proba_alpha)

sigma2_strong = 0

plt.figure(figsize=(3.5,2))
ax=plt.axes()
ax.plot([-1,0],[sigma2_neutral*L]*2,color="red")
ax.plot([0,1],[sigma2_moderate*L]*2,color="red")
ax.plot([1,2],[sigma2_strong*L]*2,color="red")
ax.yaxis.set_ticks([0,0.1,0.2])
if save:
  plt.savefig("../theta2N"+str(theta2N)+"_eta"+str(eta)+".pdf")
plt.show()

#Critical behavior
#Weak selection
gamma = np.logspace(-2,3)/L

s2N_weak = [find_s2N_weak(g,
                          theta2N,
                          eta,
                          L,
                          alphabar,
                          list_alpha,
                          list_proba_alpha,
                          a0=-10*L,b0=10*L) for g in gamma]

sigma2_weak = np.array([genetic_variance(s2N,theta2N,L,list_alpha,list_proba_alpha)
                   for s2N in s2N_weak])

plt.figure(figsize=(3.5,2))
ax=plt.axes()
ax.set_xscale("log")
ax.plot(gamma*L,sigma2_weak*L,color="red")
ax.yaxis.set_ticks([.2,.3])
if save:
  plt.savefig("../theta2N"+str(theta2N)+"_eta"+str(eta)+"_weak.pdf")
plt.show()

#Strong selection
gamma = np.logspace(-2.5,2.5)

s2N_strong = s2N_moderate #If theta is small this approximation is acceptable, else use the option below
#s2N_strong = [find_s2N_strong(g,
#                          theta2N,
#                          eta,
#                          L,
#                          alphabar,
#                          list_alpha,
#                          list_proba_alpha,
#                          a0=-10*L,b0=10*L,
#                          klim=82) for g in gamma]

sigma2_strong = np.array([genetic_variance_strong_SM(s2N_strong,
                                                  g,
                                                  theta2N,
                                                  L,
                                                  alphabar,
                                                  list_alpha,
                                                  list_proba_alpha)
                   for g in gamma])

plt.figure(figsize=(3.5,2))
ax=plt.axes()
ax.set_xscale("log")
ax.plot(gamma,sigma2_strong*L,color="red")
ax.yaxis.set_ticks([0,.2])
plt.savefig("../theta2N"+str(theta2N)+"_eta"+str(eta)+"_strong.pdf")
plt.show()

