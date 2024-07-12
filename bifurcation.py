import matplotlib.pyplot as plt
import numpy as np
from simulate_population import Population
from simulate_population import generate_pop
import scipy.special as special

#This file contains the function plot_Figurebif which draws the bifurcation diagram for 
#stabilising/disruptive selection.


N = 100 #Population size
L = 100
theta = (.1,.3) #mutation rates (forward, backwards) (.6,.9)
srange = (-15,10,100) #range of tested s values
T=20

def mean_phenotype(theta,s):
	"""Finds the mean phenotype associated with a modified beta solution
	to a WF with selection s"""
	return (2*theta[0]/(theta[0]+theta[1])
		*special.hyp1f1(2*theta[0]+1,2*(theta[0]+theta[1])+1,2*s)
		/special.hyp1f1(2*theta[0],2*(theta[0]+theta[1]),2*s)
		-1)


def match_equation(Z,s,theta,nbsteps=1000):
	"""Checks the distance between Z and the mean of pi_Z (see Theorem 5.1 of our paper)"""
	return np.abs(mean_phenotype(theta,-2*Z*s)- Z)



#Here we simulate values of equilibria by letting the system evolve for a time T at each value of s.
#N=10000

def simulate_equilibria(srange = srange,theta=theta,T=T,N=N,L=L):
	"""This function computes for each s in srange the +1 allele frequency after time T twice:
	once if we start from all +1, the other if we start from all -1"""
	equilibria =[0]*2*srange[-1]
	for (sindex,s) in enumerate(np.linspace(*srange)):
		pop=Population(theta,
				N,
				L,
				population= np.array([[False]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(s,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*sindex] = 2*np.mean(pop.population)-1
		pop=Population(theta,
				N,
				L,
				population=np.array([[True]*L]*N,dtype='bool'),
				alpha = np.ones(L))
		for t in range(int(N*T)):
			pop.selection_drift_sex(s,0,N,L)
			pop.mutation(theta,N,L)
		equilibria[2*sindex+1] = 2*np.mean(pop.population)-1
		print("Done s="+str(s))
	return np.array(equilibria)




def plot_Figurebif(srange=srange,theta=theta,T=T,N=N,L=L,equilibria=None):
	"""Plots our desired figure.
	Parameters:
	-----------
	srange: a triplet (smin,smax,nbstep) to be fed into np.linspace to get all tested values of s.

	theta: a tuple of positive floats for the mutation rates (forward, backward).

	T: how long to simulate a population before we consider that it is at equilibrium

	equilibria: if None, empirical equilibria are simulated using the function simulate_equilibria
		    Otherwise, imput a list of 2*nbsteps arrays, for instance [simulate_equilibria()]

	"""
	if equilibria is None:
		equilibria = [simulate_equilibria(srange = srange,theta=theta,T=T,N=N,L=L)]
	fig = plt.figure()
	ax=plt.axes()
	for eq in equilibria:
		ax.plot(np.repeat(np.linspace(*srange),2),eq,"or")

	list_s = np.linspace(*srange)
	list_Z = np.linspace(-1,1,N+1)
	data = np.zeros((N+1,srange[-1]))
	for (inds,s) in enumerate(list_s):
		for (indZ,Z) in enumerate(list_Z):
			data[indZ,inds] = -np.log(match_equation(Z,s,theta))

	CS = ax.contourf(list_s,list_Z,data,levels=200)
	fig.colorbar(CS)

	ax.set_xlabel("γ")
	ax.set_ylabel("Z")
	ax.set_title(
"Equilibrium value of the mean phenotype for stabilising/disruptive selection")
	plt.show()


#An accessory function for another cool plot.

def ismatched_equation(Z,s,theta,epsilon):
	"""Checks if a given Z value satisfies the nonlinear equation"""
	if np.abs(np.mean_phenotype(theta,-2*s*Z) -Z) <= epsilon:
		return True
	return False

def find_solutions(theta,epsilon=1e-3):
	list_s = np.linspace(-15,1,200)
	list_Z = np.linspace(-1,1,1001)
	list_points = []
	for s in list_s:
		for Z in list_Z:
			if ismatched_equation(Z,s,theta,epsilon):
				list_points.append([s,Z])
	return np.transpose(np.array(list_points))


def plot_bifurcations(save=False,
			list_theta = [(1.1,1.1,"green"),(0.4,.8,"red")],
			N=N,
			srange=srange,
			nbsteps=1000):
	"""Plots a bifurcation diagram showing the fixed points (s,Z) for different values of theta
	Parameters:
	-----------
	save: saves the"""
	ax = plt.axes()
	list_plots = [0]*len(list_theta)
	for (plot_index,theta) in enumerate(list_theta):
		points = find_solutions(theta[:-1],N=N,srange=srange,nbsteps=nbsteps)
		if save:
			with open("test"+str(theta)+".ndy","bw") as f:
				np.save(f,points)
		list_plots[plot_index], = ax.plot(points[0],
						points[1],
						"o",
						color=theta[-1],
						label="θ1="+str(theta[0])+", θ2="+str(theta[1]))
		print("Done: "+theta[-1])
	ax.set_ylim((-1,1))
	ax.set_xlabel("Selection strength")
	ax.set_ylabel("Z")
	ax.set_title(
"Equilibrium value of the mean phenotype for stabilising/disruptive selection")
	ax.legend(handles = list_plots)
	plt.show()

