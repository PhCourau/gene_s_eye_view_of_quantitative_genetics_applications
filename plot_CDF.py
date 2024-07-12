import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as scp
import scipy.special as special

N=2000
L=1000
omega=1000 #selection strength
eta=1/3 #selection optimum
theta=(.1,.15)
T=10

# Here we define the function plotCDFvstheory which plots the cumulative density function of
# a population at time 0 and T, and the theoretical predictions associated

def getCDF(listp):
        """Plots the cumulative distribution function of a vector of frequencies of size L"""
        L=np.shape(listp)[0]
        listy = np.repeat(np.array([k/L for k in range(L+1)]),2)
        listx = np.repeat(np.sort(listp),2)
        listx = np.append(listx,1)
        listx = np.append(0,listx)
        return (listx,listy)

def mean_phenotype_for_a_given_alpha(s,theta,alpha):
	"""	Details:
	-------
	Recall that the Laplace transform of a (alpha,beta) beta distribution is given by
	t-> 1F1(alpha,alpha+beta,t)
	In particular, we wish to compute
	integral a C(a) (2x-1)x**(2theta0 - 1) (1-x)**(2theta1-1) exp(2 a x s) dx alphadist(da)
	where C(a) is the normalization constant
	C(a) := 1/ integral x**(2theta0 - 1) (1-x)**(2theta1-1) exp(2 a x s)
	      = 1/1F1(2 theta0,2 (theta0+theta1),a)
	We are then left with
	integrate
a(
  2 theta0/(theta0+theta1)
   1F1(2 theta0 + 1, 2(theta0 + theta1) + 1,2as)/1F1(2theta0,2(theat0+theta1),2as)
	 -1) alphadist(da)
"""
	return (alpha*
(2 * theta[0]/(theta[0]+theta[1])
   * special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,2*alpha*s)
   / special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*alpha*s)
   - 1))

def mean_phenotype(s,theta,alphadist):
	"""	Details:
	-------
	Recall that the Laplace transform of a (alpha,beta) beta distribution is given by
	t-> 1F1(alpha,alpha+beta,t)
	In particular, we wish to compute
	integral a C(a) (2x-1)x**(2theta0 - 1) (1-x)**(2theta1-1) exp(2 a x s) dx alphadist(da)
	where C(a) is the normalization constant
	C(a) := 1/ integral x**(2theta0 - 1) (1-x)**(2theta1-1) exp(2 a x s)
	      = 1/1F1(2 theta0,2 (theta0+theta1),a)
	We are then left with
	integrate
a(
  2 theta0/(theta0+theta1)
   1F1(2 theta0 + 1, 2(theta0 + theta1) + 1,2as)/1F1(2theta0,2(theat0+theta1),2as)
	 -1) alphadist(da)
"""
	return scp.quad((lambda alpha:
mean_phenotype_for_a_given_alpha(s,theta,alpha)* alphadist(alpha)),
	0,
	6.5)[0]


def sqZeta(s,theta,N,alphadist,eta=eta):
	"""For a parameter s, computes the squared difference between the optimum eta and the
        mean population phenotype associated.
        Parameters
	----------
	alphadist: distribution of alphas
	"""
	return (mean_phenotype(s,theta,alphadist)-eta)**2

def sqZs(s,theta,N,alphadist,eta=eta):
	"""Here we focus on the fixed point condition: we must have
	that
	"""
	mean_ph = mean_phenotype(s,theta,alphadist)
	return (-2*omega*(mean_ph - eta) - s)**2

def gss(f, a, b, tol):
	"""Golden-section search
	to find the minimum of f on [a,b]
	f: a strictly unimodal function on [a,b]

	Example:
	>>> f = lambda x: (x - 2) ** 2
	>>> x = gss(f, 1, 5)
	>>> print("%.15f" % x)
	2.000009644875678

	This code was based on the Wikipedia article on Golden-section article
	"""
	gr = (1+np.sqrt(5))/2 #golden ratio
	while abs(b - a) > tol:
		c = b - (b - a) / gr
		d = a + (b - a) / gr
		if f(c) < f(d):  # f(c) > f(d) to find the maximum
			b = d
		else:
			a = c
	return (b + a) / 2




def plotCDFvstheory(listxalpha,
		parameters,
		alphadist,
		strong_selection=True,
		a=-100,b=100,tol=1e-4,
		step=200):
	"""Plots the cumulative distribution function.
	Parameters:
	-----------
	listxalpha: at equilibrium, a (2,L) matrix with the first row for allelic frequencies at
		each locus and the second row form the allelic effects
	parameters: contains (theta,omega,eta,N,L,T)
	alphadist: distribution of genetic effects alpha
	strong_selection: if True, then the condition on s_star is that
			  the mean trait is at the selection optimum.
			  Otherwise, the condition on s_star is that it is a
			  fixed point for which selection and mutation are
			  balanced
	a,b,tol: see gss
	step: the precision of the contour colorplot"""
	theta,omega,eta,N,L,T = parameters
	fig = plt.figure()
	ax = plt.axes()

	if strong_selection:
		s_star = gss(lambda s:sqZeta(s,theta,N,alphadist,eta),a,b,tol)
	elif omega==0:
		s_star = 0
	else:
		s_star = gss(lambda s: sqZs(s,theta,N,alphadist,eta),a,b,tol)

	print("found s_star: "+str(s_star))
	print("Theoretical mean (exact): " + str(mean_phenotype(s_star,theta,alphadist)))

	data = np.zeros((step,step))
	listx = np.linspace(1/step,1-1/step,step)
	listalpha = np.linspace(0,np.max(listxalpha[1]),step)
	for k2 in range(step):
		partition_func = special.hyp1f1(2*theta[0],
						2*(theta[0]+theta[1]),
						2*listalpha[k2]*s_star)
		for k1 in range(step):
			data[k2,k1] = (
listx[k1]**(2*theta[0]-1) * (1-listx[k1])**(2*theta[1]-1) * np.exp(2*listx[k1]*listalpha[k2]*s_star)
/partition_func * alphadist(listalpha[k2]))

	data /= np.sum(data)

	print("Theoretical mean (post discretization): "+str(listalpha @ data @ (2*listx-1)))
	print("Empirical mean: "+str(np.mean((2*listxalpha[0]-1)*listxalpha[1])))
	print("Optimum: "+str(eta))

	#data[data>10] = 10

	CS = ax.contourf(listx,
			listalpha,
			(np.log(data)+2*np.log(step))/np.log(10),
			levels=1000,
			vmin = -6, vmax = 2, #vmin=-6,vmax=1,
			extend="both")
	cbar = fig.colorbar(CS,extend="both")

	ax.plot(*listxalpha,"ro",markersize=2)

	ax.set_xlim((0,1))
	ax.set_xlabel("x")
	ax.set_ylabel("α")

	ax.set_title(
"Empirical vs Theoretical cumulative distribution function \n of allele frequencies (N="+str(N)+",L="+str(L)+",ω="+str(omega)+",η="+str(eta)[:3]+",θ="+str(theta)+")"
		)
	plt.show()
	return data,listx,listalpha
