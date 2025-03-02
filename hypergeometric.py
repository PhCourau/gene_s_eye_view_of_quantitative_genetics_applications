import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

theta = (1e-1,1e-2)

z=np.linspace(0,10,100)

f1 = special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,z)/special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),z)
f2 = special.hyp1f1(theta[0]+1,(theta[0]+theta[1])+1,z)/special.hyp1f1(theta[0],(theta[0]+theta[1]),z)
f3 = special.hyp1f1(theta[0]/5+1,(theta[0]+theta[1])/5+1,z)/special.hyp1f1(theta[0]/5,(theta[0]+theta[1])/5,z)
fth = (theta[0]+theta[1])/(theta[0]+theta[1]*np.exp(-z))

plt.plot(np.log(f1))
plt.plot(np.log(f2))
plt.plot(np.log(f3))
plt.plot(np.log(fth))
plt.show()
